//! Azure: named, and refused. Its confidential H100 VMs attest through a
//! SEV-SNP report made by Microsoft's paravisor, not by the guest for the
//! session: the client verifies a TDX quote, or a SEV-SNP report the guest
//! itself binds to the session, and neither comes from Azure. Nothing is
//! created.

use super::{
    BoxFut, Guarantee, Identity, ImageRef, ImportRequest, Instance, Leftover, Preflight, Provider,
    SerialChunk, SessionRequest, VmState,
};
use crate::quote::Quote;
use crate::Failure;

pub struct Azure;

pub fn refusal() -> Failure {
    Failure::Preflight(format!(
        "Azure is not supported: its confidential H100 VMs (Standard_NCC40ads_H100_v5) are \
         AMD SEV-SNP behind Microsoft's paravisor (guarantee {}): the report is the \
         paravisor's, not one the guest binds to this session, and this lifecycle creates \
         only the Intel TDX guests it verifies ({}). Nothing was created.",
        Guarantee::SnpViaParavisor.name(),
        Guarantee::TdxQuoteDirect.name()
    ))
}

fn refused<'a, T: 'a>() -> BoxFut<'a, T> {
    Box::pin(async { Err(refusal()) })
}

impl Provider for Azure {
    fn name(&self) -> &'static str {
        "azure"
    }
    fn display_name(&self) -> &'static str {
        "Microsoft Azure"
    }
    fn guarantee(&self) -> Guarantee {
        Guarantee::SnpViaParavisor
    }
    fn identity(&self) -> BoxFut<'_, Identity> {
        refused()
    }
    fn preflight<'a>(&'a self, _: &'a SessionRequest) -> BoxFut<'a, Preflight> {
        refused()
    }
    fn quote<'a>(&'a self, _: &'a SessionRequest, _: &'a Preflight) -> BoxFut<'a, Quote> {
        refused()
    }
    fn create<'a>(&'a self, _: &'a SessionRequest, _: &'a Instance, _: &'a [u8]) -> BoxFut<'a, ()> {
        refused()
    }
    fn instance_state<'a>(&'a self, _: &'a Instance) -> BoxFut<'a, VmState> {
        refused()
    }
    fn serial_output<'a>(&'a self, _: &'a Instance) -> BoxFut<'a, String> {
        refused()
    }
    fn serial_since<'a>(&'a self, _: &'a Instance, _: u64) -> BoxFut<'a, SerialChunk> {
        refused()
    }
    fn delete<'a>(&'a self, _: &'a Instance, _: bool) -> BoxFut<'a, ()> {
        refused()
    }
    fn sweep(&self) -> BoxFut<'_, Vec<Leftover>> {
        refused()
    }
    fn delete_leftover<'a>(&'a self, _: &'a Leftover) -> BoxFut<'a, ()> {
        refused()
    }
    fn image(&self, _: &str) -> BoxFut<'_, ImageRef> {
        refused()
    }
    fn import_image<'a>(&'a self, _: &'a ImportRequest) -> BoxFut<'a, ImageRef> {
        refused()
    }
    fn image_users<'a>(&'a self, _: &'a str) -> BoxFut<'a, Vec<String>> {
        refused()
    }
    fn delete_image<'a>(&'a self, _: &'a str) -> BoxFut<'a, ()> {
        refused()
    }
    fn manual_delete_commands(&self, _: &Instance) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_names_its_own_guarantee_and_is_not_verifiable() {
        let a = Azure;
        assert_eq!(a.guarantee(), Guarantee::SnpViaParavisor);
        assert!(!a.guarantee().verifiable());
        assert!(refusal().message().contains("SnpViaParavisor"));
        assert_eq!(refusal().code(), 91);
    }
}
