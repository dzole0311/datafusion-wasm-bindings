// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::io::Cursor;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::arrow::ipc::writer::StreamWriter;
use datafusion::arrow::util::display::FormatOptions;
use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
use datafusion::datasource::MemTable;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::disk_manager::DiskManagerConfig;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::collect;
use datafusion::sql::parser::DFParser;
use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;

use crate::console;
use crate::error::{Result, WasmError};
use crate::object_store::{OpendalRegistry, S3Config};
use crate::ResultFormat;

#[wasm_bindgen]
pub struct DataFusionContext {
    session_context: Arc<SessionContext>,
    store_registry: OpendalRegistry,
    result_format: ResultFormat,
}

#[wasm_bindgen]
impl DataFusionContext {
    pub fn greet() -> String {
        "hello from datafusion-wasm".to_string()
    }

    pub fn new() -> Self {
        crate::set_panic_hook();

        // build opendal registry
        let store_registry = OpendalRegistry::new();

        let rt = Arc::new(
            RuntimeEnvBuilder::new()
                .with_disk_manager(DiskManagerConfig::Disabled)
                .with_object_store_registry(Arc::new(store_registry.clone()))
                .build()
                .unwrap(),
        );
        let session_config = SessionConfig::new()
            .with_target_partitions(1)
            .with_information_schema(true);
        let session_context = Arc::new(SessionContext::new_with_config_rt(session_config, rt));

        console::log("datafusion context is initialized");

        Self {
            session_context,
            store_registry,
            result_format: ResultFormat::Table,
        }
    }

    pub async fn execute_sql(&self, sql: String) -> Result<String> {
        self.execute_inner(sql).await
    }

    pub async fn execute_ipc(&self, sql: String) -> Result<JsValue> {
        let (schema, record_batches) = self.collect_record_batches(sql).await?;

        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
            for batch in &record_batches {
                writer.write(batch)?;
            }
            writer.finish()?;
        }

        Ok(Uint8Array::from(buffer.as_slice()).into())
    }

    /// Run a query and register its result as a real in-memory table under
    /// `name`, replacing any existing table with that name. Unlike
    /// `CREATE TABLE ... AS` this does not go through DataFusion's DataSink
    /// write path (which spawns per-partition tasks via `collect_partitioned`
    /// and requires a live Tokio reactor); it reuses the same single-partition
    /// `collect()` path as `execute_ipc` and registers the batches directly,
    /// so it works in this Tokio-reactor-less WASM environment.
    pub async fn materialize_table(&self, name: String, query_sql: String) -> Result<()> {
        let (schema, record_batches) = self.collect_record_batches(query_sql).await?;
        let table = MemTable::try_new(schema, vec![record_batches])?;
        // Replace any existing table under this name (register_table errors
        // on a name collision instead of overwriting).
        self.session_context.deregister_table(name.as_str())?;
        self.session_context.register_table(name.as_str(), Arc::new(table))?;
        Ok(())
    }

    /// Register the record batches in an Arrow IPC stream as an in-memory
    /// table under `name`, replacing any existing table with that name.
    /// Registers the decoded batches directly as a `MemTable` (same approach
    /// as `materialize_table`), bypassing both the SQL parser and DataFusion's
    /// DataSink write path, neither of which scale/work for bulk loads in
    /// this Tokio-reactor-less WASM environment.
    pub fn register_ipc(&self, name: String, bytes: &[u8]) -> Result<()> {
        let reader = StreamReader::try_new(Cursor::new(bytes), None)?;
        let schema = reader.schema();
        let record_batches = reader.collect::<std::result::Result<Vec<_>, _>>()?;
        let table = MemTable::try_new(schema, vec![record_batches])?;
        self.session_context.deregister_table(name.as_str())?;
        self.session_context.register_table(name.as_str(), Arc::new(table))?;
        Ok(())
    }

    pub fn set_s3_config(
        &mut self,
        root: String,
        bucket: String,
        region: String,
        access_key_id: String,
        secret_access_key: String,
    ) {
        let s3_config = S3Config {
            root,
            bucket,
            region,
            access_key_id,
            secret_access_key,
        };
        self.store_registry.set_s3_config(s3_config);
    }

    pub fn set_result_format(&mut self, result_format: ResultFormat) {
        self.result_format = result_format;
    }
}

impl DataFusionContext {
    async fn execute_inner(&self, sql: String) -> Result<String> {
        let statement_batches = self.collect_statement_batches(sql).await?;
        let mut results = Vec::with_capacity(statement_batches.len());

        for (_, record_batches) in statement_batches {
            let formatted =
                pretty_format_batches_with_options(&record_batches, &FormatOptions::default())?
                    .to_string();

            results.push(formatted)
        }

        Ok(format!("{}", results.join("\n")))
    }

    async fn collect_record_batches(&self, sql: String) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let statement_batches = self.collect_statement_batches(sql).await?;
        if statement_batches.len() > 1 {
            return Err(WasmError::Other(
                "execute_ipc expects a single SQL statement".to_string(),
            ));
        }
        statement_batches
            .into_iter()
            .next()
            .ok_or_else(|| WasmError::Other("execute_ipc expects a SQL statement".to_string()))
    }

    async fn collect_statement_batches(
        &self,
        sql: String,
    ) -> Result<Vec<(SchemaRef, Vec<RecordBatch>)>> {
        let statements = DFParser::parse_sql(&sql)?;
        let mut results = Vec::with_capacity(statements.len());

        for statement in statements {
            let logical_plan = self
                .session_context
                .state()
                .statement_to_plan(statement)
                .await?;
            let data_frame = self
                .session_context
                .execute_logical_plan(logical_plan)
                .await?;
            let physical_plan = data_frame.create_physical_plan().await?;
            let schema = physical_plan.schema();

            let task_ctx = self.session_context.task_ctx();
            let record_batches = collect(physical_plan, task_ctx).await?;
            results.push((schema, record_batches))
        }

        Ok(results)
    }
}
