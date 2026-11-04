//! The quote: price per hour, its source and date, and the most it can cost.
//! The ceiling is held by time: max cost = hourly price × maximum run.

use serde_json::Value;

use crate::provider::Shape;

/// Compute Engine in the Cloud Billing catalog, read at `v1/services/<id>/skus` (provider/gcp.rs).
pub const COMPUTE_SERVICE_ID: &str = "6F81-5844-456A";

/// Used when the catalog cannot be read; printed with its date.
pub const TABLE_DATE: &str = "2026-10-16";
pub const TABLE_ORIGIN: &str = "spot prices read 20:14 UTC";
/// (machine type, region, spot USD per hour for the whole VM, without the
/// confidential surcharge). One row per region of `TDX_H100_ZONES`.
pub const TABLE: &[(&str, &str, f64)] = &[
    ("a3-highgpu-1g", "us-central1", 6.60),
    ("a3-highgpu-1g", "europe-west4", 6.26),
    ("a3-highgpu-1g", "us-east5", 3.16),
];
/// Confidential Computing on an A3 VM, billed per VM hour on top of the VM.
/// Spot 0.4391592 USD/h in every region of the page on 2026-10-16 (DWS
/// flex-start 0.512173102, unused here). No Billing SKU matched it on 2026-10-14.
pub const CC_SURCHARGE_DATE: &str = "2026-10-16";
pub const CC_SURCHARGE_SOURCE: &str =
    "https://cloud.google.com/confidential-computing/confidential-vm/pricing";
const CC_SURCHARGE: &[(&str, f64)] = &[("a3-highgpu-1g", 0.4391592)];

/// Spot confidential surcharge per hour; `None` for a machine type the page did not list.
pub fn cc_surcharge(machine_type: &str) -> Option<f64> {
    CC_SURCHARGE
        .iter()
        .find(|(m, _)| *m == machine_type)
        .map(|(_, p)| *p)
}

/// External IP and boot disk, per hour: an estimate, not read from the catalog.
pub const EXTRAS_USD_PER_HOUR: f64 = 0.01;

#[derive(Debug, Clone, PartialEq)]
pub enum PriceSource {
    Catalog {
        read_at_unix: u64,
    },
    Table {
        date: &'static str,
        origin: &'static str,
    },
}

#[derive(Debug, Clone)]
pub struct Quote {
    pub machine_type: String,
    pub region: String,
    pub vm_usd_per_hour: f64,
    pub cc_usd_per_hour: f64,
    pub extras_usd_per_hour: f64,
    pub max_run_secs: u64,
    pub source: PriceSource,
    /// Catalog only: GPU, vCPU and RAM parts.
    pub breakdown: Option<Breakdown>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Breakdown {
    pub gpu_usd_per_hour: f64,
    pub vcpu_usd_per_hour: f64,
    pub ram_usd_per_gib_hour: f64,
}

impl Quote {
    pub fn hourly(&self) -> f64 {
        self.vm_usd_per_hour + self.cc_usd_per_hour + self.extras_usd_per_hour
    }

    /// Rounded up to the cent: what the user agrees to at most.
    pub fn max_cost(&self) -> f64 {
        ceil_cents(self.hourly() * self.max_run_secs as f64 / 3600.0)
    }

    pub fn cost_after(&self, secs: u64) -> f64 {
        ceil_cents(self.hourly() * secs.min(self.max_run_secs) as f64 / 3600.0)
    }

    pub fn source_line(&self) -> String {
        match &self.source {
            PriceSource::Catalog { read_at_unix } => format!(
                "Cloud Billing catalog, {}",
                crate::datetime_utc(*read_at_unix)
            ),
            PriceSource::Table { date, origin } => format!(
                "embedded table dated {date} ({origin}); the catalog could not be read, prices may have moved"
            ),
        }
    }

    /// `now`: when the VM would start; its deletion time is printed from it.
    pub fn render(&self, gpu_label: &str, now: u64) -> String {
        self.render_styled(gpu_label, now, None, false)
    }

    /// `render`, colored when `on`: the heading bold, the prices, the deletion
    /// time, the maximum and the spot warning yellow, the source of the surcharge dim.
    /// The deletion time is never after `end_by`, set by a token `up` cannot renew.
    pub fn render_styled(
        &self,
        gpu_label: &str,
        now: u64,
        end_by: Option<u64>,
        on: bool,
    ) -> String {
        use crate::color::{paint, Style};
        let mins = self.max_run_secs / 60;
        let ends = now + self.max_run_secs;
        let deletes_at = end_by.map_or(ends, |b| b.min(ends));
        let price = |label: &str, usd: f64, unit: &str| {
            format!(
                "  {label:<46}{}\n",
                paint(on, Style::Warn, &format!("{usd:>6.2} {unit}"))
            )
        };
        let mut s = paint(
            on,
            Style::Bold,
            &format!(
                "Quote ({}; estimate, billed by Google to your project)",
                self.source_line()
            ),
        );
        s.push('\n');
        s.push_str(&price(
            &format!("{} spot, {gpu_label}, Intel TDX", self.machine_type),
            self.vm_usd_per_hour,
            "USD/h",
        ));
        s.push_str(&price(
            "Confidential Computing on A3, spot",
            self.cc_usd_per_hour,
            "USD/h",
        ));
        s.push_str(&format!(
            "    {}\n",
            paint(
                on,
                Style::Dim,
                &format!(
                    "(Google's pricing page, read {CC_SURCHARGE_DATE}: {CC_SURCHARGE_SOURCE})"
                )
            )
        ));
        s.push_str(&price(
            "external IP, boot disk",
            self.extras_usd_per_hour,
            "USD/h",
        ));
        s.push_str(&format!(
            "  {}\n",
            paint(
                on,
                Style::Warn,
                &format!(
                    "the VM deletes itself at {} ({mins} min)",
                    crate::hhmm_utc(deletes_at)
                )
            )
        ));
        s.push_str(&format!(
            "  {}\n",
            paint(
                on,
                Style::WarnBold,
                &format!(
                    "{:<46}{:>6.2} USD",
                    "maximum cost if it runs to the end",
                    self.max_cost()
                )
            )
        ));
        s.push_str(&format!(
            "  {}\n",
            paint(
                on,
                Style::Warn,
                "Spot: Google may stop it at any time; you pay only the time used."
            )
        ));
        s
    }
}

pub fn ceil_cents(usd: f64) -> f64 {
    (usd * 100.0 - 1e-9).ceil().max(0.0) / 100.0
}

pub fn table_price(machine_type: &str, region: &str) -> Option<f64> {
    TABLE
        .iter()
        .find(|(m, r, _)| *m == machine_type && *r == region)
        .map(|(_, _, p)| *p)
}

/// USD of the last tier of a SKU's first pricing expression.
fn unit_price(sku: &Value) -> Option<(f64, String)> {
    let pe = &sku["pricingInfo"][0]["pricingExpression"];
    let rate = pe["tieredRates"].as_array()?.last()?;
    let up = &rate["unitPrice"];
    if up["currencyCode"].as_str().unwrap_or("USD") != "USD" {
        return None;
    }
    let units: f64 = match &up["units"] {
        Value::String(s) => s.parse().ok()?,
        Value::Number(n) => n.as_f64()?,
        Value::Null => 0.0,
        _ => return None,
    };
    let nanos = up["nanos"].as_f64().unwrap_or(0.0);
    Some((
        units + nanos / 1e9,
        pe["usageUnit"].as_str().unwrap_or("").to_string(),
    ))
}

/// Spot H100 GPU, A3 vCPU and A3 RAM for a region, from catalog SKUs.
/// Descriptions are matched loosely (unverified): the exact strings were not
/// recorded when the catalog was read on 2026-10-05.
pub fn breakdown_from_skus(skus: &[Value], region: &str) -> Option<Breakdown> {
    let mut gpu = None;
    let mut cpu = None;
    let mut ram = None;
    for sku in skus {
        let desc = sku["description"]
            .as_str()
            .unwrap_or("")
            .to_ascii_lowercase();
        let usage = sku["category"]["usageType"].as_str().unwrap_or("");
        let in_region = sku["serviceRegions"]
            .as_array()
            .is_some_and(|rs| rs.iter().any(|r| r.as_str() == Some(region)));
        if usage != "Preemptible" || !in_region {
            continue;
        }
        if [
            "mega", "ultra", "edge", "plus", "commit", "reserved", "dws", "calendar", "flex",
        ]
        .iter()
        .any(|w| desc.contains(w))
        {
            continue;
        }
        let Some((price, unit)) = unit_price(sku) else {
            continue;
        };
        if desc.contains("h100 80gb") && unit == "h" {
            gpu.get_or_insert(price);
        } else if desc.contains("a3 instance core") && unit == "h" {
            cpu.get_or_insert(price);
        } else if desc.contains("a3 instance ram") && unit.starts_with("GiBy") {
            ram.get_or_insert(price);
        }
    }
    Some(Breakdown {
        gpu_usd_per_hour: gpu?,
        vcpu_usd_per_hour: cpu?,
        ram_usd_per_gib_hour: ram?,
    })
}

pub fn vm_price(b: &Breakdown, shape: &Shape) -> f64 {
    b.gpu_usd_per_hour * f64::from(shape.gpus)
        + b.vcpu_usd_per_hour * f64::from(shape.vcpus)
        + b.ram_usd_per_gib_hour * shape.memory_gib
}

/// Keeps only the SKUs `breakdown_from_skus` may use (pages are megabytes).
pub fn is_candidate(sku: &Value) -> bool {
    let d = sku["description"].as_str().unwrap_or("");
    d.contains("H100") || d.contains("A3 Instance")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sku(desc: &str, usage: &str, region: &str, unit: &str, units: &str, nanos: i64) -> Value {
        json!({
            "description": desc,
            "category": {"usageType": usage, "resourceFamily": "Compute"},
            "serviceRegions": [region],
            "pricingInfo": [{"pricingExpression": {"usageUnit": unit,
                "tieredRates": [{"startUsageAmount": 0,
                  "unitPrice": {"currencyCode": "USD", "units": units, "nanos": nanos}}]}}]
        })
    }

    #[test]
    fn the_catalog_breakdown_ignores_other_regions_usages_and_shapes() {
        let skus = vec![
            sku(
                "Spot Preemptible Nvidia H100 80GB GPU running in Americas",
                "Preemptible",
                "us-central1",
                "h",
                "5",
                570_000_000,
            ),
            sku(
                "Nvidia H100 80GB GPU running in Americas",
                "OnDemand",
                "us-central1",
                "h",
                "11",
                0,
            ),
            sku(
                "Spot Preemptible Nvidia H100 80GB Mega GPU running in Americas",
                "Preemptible",
                "us-central1",
                "h",
                "9",
                0,
            ),
            sku(
                "Spot Preemptible Nvidia H100 80GB GPU running in Netherlands",
                "Preemptible",
                "europe-west4",
                "h",
                "7",
                0,
            ),
            sku(
                "Spot Preemptible A3 Instance Core running in Americas",
                "Preemptible",
                "us-central1",
                "h",
                "0",
                10_000_000,
            ),
            sku(
                "Spot Preemptible A3 Instance Ram running in Americas",
                "Preemptible",
                "us-central1",
                "GiBy.h",
                "0",
                1_000_000,
            ),
        ];
        let b = breakdown_from_skus(&skus, "us-central1").unwrap();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(b.gpu_usd_per_hour, 5.57), "{b:?}");
        assert!(close(b.vcpu_usd_per_hour, 0.01), "{b:?}");
        assert!(close(b.ram_usd_per_gib_hour, 0.001), "{b:?}");
        let shape = Shape {
            vcpus: 26,
            memory_gib: 234.0,
            gpus: 1,
        };
        let p = vm_price(&b, &shape);
        assert!((p - (5.57 + 0.26 + 0.234)).abs() < 1e-9, "{p}");
        assert!(breakdown_from_skus(&skus, "us-east5").is_none());
    }

    #[test]
    fn the_maximum_cost_is_the_hourly_price_over_the_whole_run() {
        let q = Quote {
            machine_type: "a3-highgpu-1g".into(),
            region: "us-central1".into(),
            vm_usd_per_hour: 6.29,
            cc_usd_per_hour: 0.4391592,
            extras_usd_per_hour: EXTRAS_USD_PER_HOUR,
            max_run_secs: 3600,
            source: PriceSource::Table {
                date: TABLE_DATE,
                origin: TABLE_ORIGIN,
            },
            breakdown: None,
        };
        // 6.29 + 0.4391592 + 0.01 = 6.7391592 USD/h: the surcharge counts in the ceiling.
        assert_eq!(q.max_cost(), 6.74);
        assert_eq!(q.cost_after(1800), 3.37);
        assert_eq!(q.cost_after(10 * 3600), 6.74);
        let r = q.render("1x H100 80GB", 1_791_839_109);
        assert!(r.contains("embedded table dated 2026-10-16"), "{r}");
        assert!(r.contains("Confidential Computing on A3, spot"), "{r}");
        assert!(r.contains("  0.44 USD/h"), "{r}");
        assert!(r.contains(CC_SURCHARGE_SOURCE), "{r}");
        assert!(r.contains("read 2026-10-16"), "{r}");
        assert!(
            r.contains("the VM deletes itself at 22:05 UTC (60 min)"),
            "{r}"
        );
        assert!(r.contains("6.74 USD"), "{r}");
        // Colored, the same text; the maximum in bold yellow, nothing else changed.
        let c = q.render_styled("1x H100 80GB", 1_791_839_109, None, true);
        assert_eq!(crate::color::strip(&c), r);
        assert!(
            c.contains("\x1b[1;33mmaximum cost if it runs to the end"),
            "{c:?}"
        );
        assert!(c.contains("\x1b[33m  0.44 USD/h\x1b[0m\n"), "{c:?}");
        assert_eq!(
            q.render_styled("1x H100 80GB", 1_791_839_109, None, false),
            r
        );
    }

    /// A cut computed 30 s before the quote is shown: 3311 s from 21:04:39 end
    /// by 21:59:50, and 3311 s from 21:05:09 would print 22:00.
    #[test]
    fn the_deletion_time_is_never_later_than_the_end_a_token_sets() {
        let q = Quote {
            machine_type: "a3-highgpu-1g".into(),
            region: "us-central1".into(),
            vm_usd_per_hour: 6.29,
            cc_usd_per_hour: 0.4391592,
            extras_usd_per_hour: EXTRAS_USD_PER_HOUR,
            max_run_secs: 3311,
            source: PriceSource::Table {
                date: TABLE_DATE,
                origin: TABLE_ORIGIN,
            },
            breakdown: None,
        };
        // 21:05:09 UTC.
        let now = 1_791_839_109;
        let end_by = now - 30 + 3311;
        let r = q.render_styled("1x H100 80GB", now, Some(end_by), false);
        assert!(
            r.contains("the VM deletes itself at 21:59 UTC (55 min)"),
            "{r}"
        );
        assert!(
            q.render("1x H100 80GB", now)
                .contains("the VM deletes itself at 22:00 UTC (55 min)"),
            "without an end"
        );
        // An end after now + max_run changes nothing.
        assert_eq!(
            q.render_styled("1x H100 80GB", now, Some(now + 3311 + 60), false),
            q.render("1x H100 80GB", now)
        );
    }

    /// Google's Confidential VM pricing page, read 2026-10-16 22:45 UTC: the same
    /// figure in every region it lists, and no Cloud Billing SKU found for it (14/10).
    #[test]
    fn the_confidential_surcharge_is_dated_and_sourced() {
        assert_eq!(cc_surcharge("a3-highgpu-1g"), Some(0.4391592));
        assert_eq!(cc_surcharge("n2-standard-2"), None);
        assert_eq!(CC_SURCHARGE_DATE, "2026-10-16");
        assert!(CC_SURCHARGE_SOURCE.starts_with("https://cloud.google.com/"));
    }

    /// Spot a3-highgpu-1g read 2026-10-16 20:14 UTC; every zone `up` may pick has a price.
    #[test]
    fn the_table_is_dated_and_regional() {
        assert_eq!(TABLE_DATE, "2026-10-16");
        assert_eq!(table_price("a3-highgpu-1g", "us-central1"), Some(6.60));
        assert_eq!(table_price("a3-highgpu-1g", "europe-west4"), Some(6.26));
        assert_eq!(table_price("a3-highgpu-1g", "us-east5"), Some(3.16));
        for zone in crate::TDX_H100_ZONES {
            let region = crate::region_of(zone);
            assert!(table_price(crate::MACHINE_TYPE, region).is_some(), "{zone}");
        }
        assert_eq!(table_price("a3-highgpu-1g", "asia-east1"), None);
    }
}
