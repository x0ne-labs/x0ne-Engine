#!/usr/bin/env bash
# Pins the transparency log of image references in this tree, after the
# archive key (scripts/pin-archive-key.sh): sets transparency::RELEASE from a
# pin file, replaces the test that wanted no pin, and rewrites every text that
# says this tree pins no log: README.md, docs/cloud.md, docs/attested-channel.md,
# docs/pinning.md, x0ne-infer/THREAT_MODEL.md; CHANGELOG.md gets a new Unreleased
# entry, its released entries are left as written. Nothing is written unless
# every check passes and every text is found.
#
# Usage: scripts/pin-transparency-log.sh <log.pin>
#   PIN_DATE=YYYY-MM-DD (default: today, UTC) dates the pin in the texts.
# The pin file, one value per line, '#' comments:
#   origin <the log's origin line>
#   log <the log's verifier key, Ed25519 (type 0x01), named by the origin>
#   witness <a witness's cosignature/v1 verifier key (type 0x04)>   (one or more)
#   quorum <how many distinct witnesses must cosign, 1 to their number>
#   max-age <seconds a cosignature counts, 1 to 31536000>
#   tiles <https:// location of the log's checkpoint and tiles>
# Refuses: a key ID that is not the key's, a log key named otherwise or of
# another type, a witness twice or two names for one key, a quorum or age out
# of range, a location that is not plain https, an origin or witness that says
# TEST or names test.invalid, one that says REHEARSAL (any case) unless
# ALLOW_REHEARSAL_LOG=1 (a rehearsal, in a throwaway clone), a tree whose
# archive key is not pinned yet, a tree that already pins a log.
# Prints the SHA-256 fingerprint of each key, as written on paper when it was
# made (x0ne-engine-tlog keygen). Then: make ci, read the diff, commit.
# Needs bash and perl (Digest::SHA, MIME::Base64).
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <log.pin>" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec perl -w - "$ROOT" "$1" "${PIN_DATE:-$(date -u +%F)}" <<'PERL'
use strict;
use Digest::SHA qw(sha256 sha256_hex);
use MIME::Base64 qw(decode_base64 encode_base64);
my ($root, $pinfile, $date) = @ARGV;
sub refuse { print STDERR "pin-transparency-log: $_[0]; nothing written\n"; exit 1 }
$date =~ /^\d{4}-\d{2}-\d{2}$/ or refuse("PIN_DATE $date: expected YYYY-MM-DD");

open my $ph, '<', $pinfile or refuse("$pinfile: $!");
my $pin = do { local $/; <$ph> };
close $ph;
defined $pin && length $pin or refuse("$pinfile is empty");
length $pin <= 16384 or refuse("$pinfile: over 16 KiB");
my (%one, @wit);
my $n = 0;
for my $l (split /\n/, $pin) {
    $n++;
    next if $l =~ /^\s*(#|$)/;
    my ($k, $v) = $l =~ /^(origin|log|witness|quorum|max-age|tiles) (\S+)$/
        or refuse("$pinfile line $n: unknown line \"$l\"");
    if ($k eq 'witness') { push @wit, $v; next }
    exists $one{$k} and refuse("$pinfile: $k twice");
    $one{$k} = $v;
}
exists $one{$_} or refuse("$pinfile: no $_ line") for qw(origin log quorum max-age tiles);
@wit or refuse("$pinfile: no witness line, at least one witness is required");

# A name as signed-note and a Rust string literal take it.
my $NAME = qr{^[A-Za-z0-9._~:/-]{1,255}$};
sub says_test {
    my ($what, $s) = @_;
    ($s =~ /TEST/ || $s =~ /test\.invalid/i)
        and refuse("$what \"$s\" says TEST: the TEST log's keys are derived from public strings");
}
my @rehearsal;
sub vkey {
    my ($what, $v) = @_;
    my ($name, $id, $b64) = $v =~ /^([^+\s]+)\+([^+]*)\+(.+)$/
        or refuse("$what \"$v\": not <name>+<key ID>+<key>");
    $name =~ $NAME or refuse("$what \"$name\": a name of letters, digits and ._~:/- only");
    says_test($what, $name);
    push @rehearsal, $name if $name =~ /rehearsal/i;
    $id =~ /^[0-9a-f]{8}$/ or refuse("$what $name: key ID \"$id\" is not 8 lowercase hex");
    $b64 =~ m{^[A-Za-z0-9+/]+=*$} or refuse("$what $name: the key is not base64");
    my $raw = decode_base64($b64);
    encode_base64($raw, '') eq $b64 or refuse("$what $name: the key is not canonical base64");
    length $raw == 33 or refuse("$what $name: " . length($raw) . " key bytes, expected 33");
    my $want = substr sha256_hex("$name\n$raw"), 0, 8;
    $want eq $id or refuse("$what $name: key ID $id is not the key's ($want)");
    return ($name, ord(substr $raw, 0, 1), substr($raw, 1));
}
sub fingerprint {
    my @h = unpack '(A4)*', uc substr(sha256_hex($_[0]), 0, 40);
    return "@h";
}

my $origin = $one{origin};
$origin =~ $NAME or refuse("origin \"$origin\": letters, digits and ._~:/- only");
says_test('origin', $origin);
push @rehearsal, $origin if $origin =~ /rehearsal/i;
my ($lname, $ltype, $lkey) = vkey('log key', $one{log});
$lname eq $origin
    or refuse("log key is named \"$lname\", the origin is \"$origin\": a log key is named by its origin");
$ltype == 0x01 or refuse(sprintf "log key %s: type 0x%02x, a log signs with Ed25519 (0x01)", $lname, $ltype);
my (%wname, %wkey);
$wkey{$lkey} = 'the log key';
for my $w (@wit) {
    my ($name, $type, $key) = vkey('witness', $w);
    $type == 0x04 or refuse(sprintf "witness %s: type 0x%02x, not a cosignature/v1 key (0x04)", $name, $type);
    $wname{$name}++ and refuse("witness $name twice");
    exists $wkey{$key} and refuse("witness $name: the same key as $wkey{$key}");
    $wkey{$key} = $name;
}
my $q = $one{quorum};
($q =~ /^[1-9][0-9]{0,2}$/ && $q <= @wit) or refuse("quorum $q: between 1 and the " . @wit . " witnesses");
my $age = $one{'max-age'};
($age =~ /^[1-9][0-9]{0,7}$/ && $age <= 31536000) or refuse("max-age $age: seconds, between 1 and 31536000 (a year)");
my $tiles = $one{tiles};
$tiles =~ m{^https://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?$}
    or refuse("tiles $tiles: an https:// location, without credentials, query or fragment");
my $allow = ($ENV{ALLOW_REHEARSAL_LOG} // '') eq '1';
@rehearsal and !$allow
    and refuse("\"$rehearsal[0]\" says REHEARSAL: never the release pin; ALLOW_REHEARSAL_LOG=1 pins it, in a throwaway clone");

# Edits, in memory; each anchor must be found exactly once.
my (%new, %old);
sub file {
    my $rel = shift;
    return $new{$rel} if exists $new{$rel};
    open my $h, '<:raw', "$root/$rel" or refuse("$rel: $!");
    local $/;
    return $old{$rel} = $new{$rel} = <$h>;
}
sub swap {
    my ($rel, $old, $rep) = @_;
    my $s = file($rel);
    my $c = () = $s =~ /\Q$old\E/g;
    $c == 1 or refuse("$rel: the text to replace is there $c times, not once: \"" . substr($old, 0, 60) . "...\"");
    my $at = index $s, $old;
    substr($s, $at, length $old) = $rep;
    $new{$rel} = $s;
}
# A pattern with the archive pin's fingerprint and date in $1 and $2.
sub rswap {
    my ($rel, $re, $rep) = @_;
    my $s = file($rel);
    # Count matches, not captures: a pattern with groups returns them all in list context.
    my $c = 0;
    $c++ while $s =~ /$re/g;
    $c == 1 or refuse("$rel: the text to replace is there $c times, not once: \"" . substr("$re", 0, 80) . "...\"");
    $s =~ s/$re/$rep->($1, $2)/e;
    $new{$rel} = $s;
}
file('x0ne-infer/src/archive_key.rs') =~ /^pub const RELEASE: Option<Pin> = None;$/m
    and refuse('x0ne-infer/src/archive_key.rs pins no archive key: pin the archive key first (scripts/pin-archive-key.sh)');
file('x0ne-infer/src/transparency.rs') =~ /^pub const RELEASE: Option<Pin> = None;$/m
    or refuse('x0ne-infer/src/transparency.rs: RELEASE is not None, this tree already pins a log');

my $nw = @wit;
my $wlist = join ",\n", map { "`$_`" } @wit;
my $wrust = join '', map { "        \"$_\",\n" } @wit;
my $FPR = qr/`[0-9A-F]{40}`/;
# rustfmt (width 100): a field and its string share a line when they fit, else
# the string goes on the next line. Past 100 there too, rustfmt keeps the text.
sub rfield {
    my ($name, $value) = @_;
    my $line = "    $name: \"$value\",";
    return length($line) <= 100 ? "$line\n" : "    $name:\n        \"$value\",\n";
}
my ($rorigin, $rlog, $rtiles) = (rfield('origin', $origin), rfield('log_key', $one{log}), rfield('tiles', $tiles));
my $DAY = qr/\d{4}-\d{2}-\d{2}/;

swap('x0ne-infer/src/transparency.rs', <<'OLD', <<"NEW");
//! the archive key (`archive_key.rs`): `RELEASE` is None until x0ne runs a
//! log and witnesses agree to cosign it, and until then every reference is
//! refused; a debug build with `test-archive-key` pins the TEST log
OLD
//! the archive key (`archive_key.rs`): `RELEASE` names the log pinned on
//! $date (`scripts/pin-transparency-log.sh`); a build made before that
//! pins none and refuses every reference; a debug build with
//! `test-archive-key` pins the TEST log
NEW
swap('x0ne-infer/src/transparency.rs', <<'OLD', <<"NEW");
/// The release pin: None until x0ne runs a public log with witnesses.
pub const RELEASE: Option<Pin> = None;
OLD
/// The release pin of the transparency log, written on $date by
/// scripts/pin-transparency-log.sh. A tree made before it had None.
pub const RELEASE: Option<Pin> = Some(Pin {
$rorigin$rlog    witnesses: &[
$wrust    ],
    quorum: $q,
    max_age_s: $age,
$rtiles});
NEW
# rustfmt: a macro call whose arguments pass 60 columns goes one per line.
my $origin_eq = length("pin.origin, \"$origin\"") > 60
    ? "        assert_eq!(\n            pin.origin,\n            \"$origin\"\n        );\n"
    : "        assert_eq!(pin.origin, \"$origin\");\n";
swap('x0ne-infer/src/transparency.rs', <<'OLD', <<"NEW");
    /// The release pin: none yet, and every reference is refused saying why.
    /// A pin without a witness, with a log key of another name, or with
    /// a plain HTTP location is refused.
    #[test]
    fn no_release_log_refuses_everything_and_pins_are_checked() {
        assert!(
            RELEASE.is_none(),
            "a pin landed: replace this test by one on it"
        );
OLD
    /// The pin of $date passes `pinned` and is not the TEST log; a build
    /// without one refuses every reference saying why. A pin without a
    /// witness, with a log key of another name, or with a plain HTTP
    /// location is refused.
    #[test]
    fn the_release_pin_is_the_ceremony_log() {
        let Some(pin) = &RELEASE else {
            panic!("RELEASE is None")
        };
${origin_eq}        let release = pinned(pin).expect("the pin is the log written down");
        assert_ne!(release.log, log_key(), "the TEST log key");
        assert!(
            !release.witnesses.contains(&witness_key()),
            "the TEST witness key"
        );
NEW
rswap('README.md',
    qr/relay pins the x0ne archive key with primary ($FPR) since ($DAY),\nbut no transparency log \(exit 100\), and no image is published\.\n/,
    sub { "relay pins the x0ne archive key with primary $_[0] since $_[1]\n"
        . "and the transparency log `$origin` since $date, but no image is published.\n" });
rswap('docs/attested-channel.md',
    qr/> pins the x0ne archive key, primary ($FPR), since ($DAY);\n> it refuses every image reference with 100, as\n> it pins no transparency log\. No `x0ne-cvm` image is published, and\n/,
    sub { "> pins the x0ne archive key, primary $_[0], since $_[1],\n"
        . "> and the transparency log `$origin` since $date. No `x0ne-cvm` image is\n"
        . "> published, and\n" });
rswap('docs/cloud.md',
    qr/> pins the x0ne archive key, primary ($FPR), since ($DAY),\n> but it pins no transparency log, so it refuses every image reference\n> with 100\. `up` asks the relay first and stops there,\n> before the token\. No `x0ne-cvm` image is published to import:\n/,
    sub { "> pins the x0ne archive key, primary $_[0], since $_[1],\n"
        . "> and the transparency log `$origin` since $date. No `x0ne-cvm` image is\n"
        . "> published to import:\n" });
swap('docs/cloud.md', <<'OLD', <<"NEW");
does not sign its cosigner's name. **This source tree pins no log**:
`RELEASE` is `None` and a release relay refuses every reference with 100, as
it refuses them with 99 while no archive key is pinned. x0ne runs no public
log and no witness yet. A pinned log is read at its pinned `https://` location
OLD
does not sign its cosigner's name. **This source tree pins the log** `$origin`
since $date (`scripts/pin-transparency-log.sh <log.pin>`): log key
`$one{log}`; witnesses
$wlist;
quorum $q, maximum age $age s, tiles at `$tiles`. A release build made
before that pins no log and refuses every reference with 100, as it refuses
them with 99 while no archive key is pinned. A pinned log is read at its pinned `https://` location
NEW
swap('docs/pinning.md',
    '**Transparency log of image references (built, no log runs yet).**',
    "**Transparency log of image references (pinned since $date).**");
swap('docs/pinning.md', <<'OLD', <<"NEW");
alone visible to everyone; it does not make it good. This tree pins no log:
x0ne runs no public log and no witness yet, and a release relay refuses
every reference until one is pinned. It was tested against a local TEST log
OLD
alone visible to everyone; it does not make it good. This tree pins the
log `$origin` since $date ($nw witnesses, quorum $q); a release
build made before that refuses every reference. It was tested against a local TEST log
NEW
swap('x0ne-infer/THREAT_MODEL.md',
    '**This source tree pins no log** (`transparency::RELEASE` is `None`): a release build refuses every reference with 100, and x0ne runs no log and no witness yet.',
    "**This source tree pins the log** `$origin` since $date (`transparency::RELEASE`, $nw witnesses, quorum $q): a release build made before that refuses every reference with 100.");
swap('x0ne-infer/THREAT_MODEL.md',
    'done on a TEST log, 2026-10-22; no release log exists).',
    "done on a TEST log, 2026-10-22; the release log pinned on $date).");
# Released entries are history: the pin is a new Unreleased entry.
swap('CHANGELOG.md', "## Unreleased\n\n", <<"NEW");
## Unreleased

- The transparency log `$origin` is pinned since $date
  (`scripts/pin-transparency-log.sh`): $nw witnesses, quorum $q, maximum age
  $age s, tiles at `$tiles`. A release build made before that pins none
  and refuses every reference with 100.
NEW

# Write: each file through a temporary and a rename; a file read for a check
# only is left as it is, and not named.
for my $rel (grep { $new{$_} ne $old{$_} } sort keys %new) {
    my $t = "$root/$rel.pin-tmp";
    open my $h, '>:raw', $t or die "$t: $!";
    print $h $new{$rel} or die "$t: $!";
    close $h or die "$t: $!";
    chmod((stat "$root/$rel")[2] & 07777, $t);
    rename $t, "$root/$rel" or die "$rel: $!";
    print "pinned: $rel\n";
}
print "pin-transparency-log: ALLOW_REHEARSAL_LOG=1: REHEARSAL log pinned, this tree is a rehearsal, never a release\n"
    if @rehearsal;
print "pin-transparency-log: origin $origin, $nw witnesses, quorum $q, max-age $age s, tiles $tiles, $date\n";
print "log key fingerprint ", fingerprint($one{log}), "\n";
print "witness $_ fingerprint ", fingerprint($_), "\n" for @wit;
print "Compare each fingerprint with the paper written when the key was made. Next: make ci, git diff, commit.\n";
PERL
