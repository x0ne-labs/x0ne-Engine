//! The OpenAI-compatible contract the chat surfaces share: the TD served
//! directly (`server`), the cloud TD (`cloud_serve`) and the relay
//! (`local_api`). Each surface keeps its own handler, since each stands
//! in its own place of trust; what a request may say lives here, once.

pub mod error;
pub mod models;
pub mod request;
