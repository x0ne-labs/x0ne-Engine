// Interop between x0ne-infer's tlog.rs / transparency.rs and the Go
// implementations of the same C2SP formats (golang.org/x/mod/sumdb/note and
// tlog, transparency-dev/formats and merkle). Local files only.
//
//	interop make <dir>    write a Go-made log, checkpoints, tiles and proofs
//	interop verify <dir>  verify a Rust-made TEST log with the Go libraries
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	flog "github.com/transparency-dev/formats/log"
	fnote "github.com/transparency-dev/formats/note"
	"github.com/transparency-dev/merkle/compact"
	"github.com/transparency-dev/merkle/proof"
	"github.com/transparency-dev/merkle/rfc6962"
	"golang.org/x/mod/sumdb/note"
	"golang.org/x/mod/sumdb/tlog"
)

const (
	goOrigin   = "interop.invalid/go-log"
	goWitness  = "interop.invalid/go-witness"
	entryHead  = "x0ne-engine/image-reference/v1"
	proofHead  = "c2sp.org/tlog-proof@v1"
	tileWidth  = 256
	nEntries   = 600
	refIndex   = 300
	firstSize  = 301
)

func die(f string, a ...any) { fmt.Fprintf(os.Stderr, "FAIL: "+f+"\n", a...); os.Exit(1) }

func keyHash(name string, algKey []byte) uint32 {
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte("\n"))
	h.Write(algKey)
	return binary.BigEndian.Uint32(h.Sum(nil)[:4])
}

func seed(label string) []byte { s := sha256.Sum256([]byte(label)); return s[:] }

func skey(name string, alg byte, sd []byte) string {
	pub := ed25519.NewKeyFromSeed(sd).Public().(ed25519.PublicKey)
	return fmt.Sprintf("PRIVATE+KEY+%s+%08x+%s", name, keyHash(name, append([]byte{alg}, pub...)),
		base64.StdEncoding.EncodeToString(append([]byte{alg}, sd...)))
}

func vkey(name string, alg byte, sd []byte) string {
	pub := ed25519.NewKeyFromSeed(sd).Public().(ed25519.PublicKey)
	k := append([]byte{alg}, pub...)
	return fmt.Sprintf("%s+%08x+%s", name, keyHash(name, k), base64.StdEncoding.EncodeToString(k))
}

func entry(ref []byte) []byte {
	d := sha256.Sum256(ref)
	return []byte(fmt.Sprintf("%s\nsha256 %s\n", entryHead, hex.EncodeToString(d[:])))
}

// C2SP tlog-tiles index: groups of three digits, all but the last prefixed x.
func tileIndex(n uint64) string {
	s := fmt.Sprintf("%03d", n%1000)
	for n /= 1000; n > 0; n /= 1000 {
		s = fmt.Sprintf("x%03d/", n%1000) + s
	}
	return s
}

func tilePath(level int, n uint64, width int) string {
	p := fmt.Sprintf("tile/%d/%s", level, tileIndex(n))
	if width < tileWidth {
		p += fmt.Sprintf(".p/%d", width)
	}
	return p
}

type store []tlog.Hash

func (s *store) ReadHashes(idx []int64) ([]tlog.Hash, error) {
	out := make([]tlog.Hash, len(idx))
	for i, x := range idx {
		if x >= int64(len(*s)) {
			return nil, fmt.Errorf("hash %d not stored", x)
		}
		out[i] = (*s)[x]
	}
	return out, nil
}

func b64lines(hs []tlog.Hash) string {
	var b strings.Builder
	for _, h := range hs {
		b.WriteString(base64.StdEncoding.EncodeToString(h[:]) + "\n")
	}
	return b.String()
}

func write(dir, name string, data []byte) {
	p := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		die("%v", err)
	}
	if err := os.WriteFile(p, data, 0o644); err != nil {
		die("%v", err)
	}
}

// Tiles of the tree of `size` leaves: full tiles and the partial ones.
func writeTiles(dir string, s *store, size int64) {
	for level := 0; ; level++ {
		// hashes at this tile level: subtrees of 256^level leaves
		span := int64(1)
		for i := 0; i < level; i++ {
			span *= tileWidth
		}
		count := size / span
		if count == 0 {
			return
		}
		for t := int64(0); t*tileWidth < count; t++ {
			w := count - t*tileWidth
			if w > tileWidth {
				w = tileWidth
			}
			var b bytes.Buffer
			for k := t * tileWidth; k < t*tileWidth+w; k++ {
				h, err := s.ReadHashes([]int64{tlog.StoredHashIndex(8*level, k)})
				if err != nil {
					die("%v", err)
				}
				b.Write(h[0][:])
			}
			write(dir, tilePath(level, uint64(t), int(w)), b.Bytes())
		}
	}
}

func checkpointNote(size int64, root tlog.Hash, signers ...note.Signer) []byte {
	text := fmt.Sprintf("%s\n%d\n%s\n", goOrigin, size, base64.StdEncoding.EncodeToString(root[:]))
	msg, err := note.Sign(&note.Note{Text: text}, signers...)
	if err != nil {
		die("sign: %v", err)
	}
	return msg
}

func makeLog(dir string) {
	logSigner, err := note.NewSigner(skey(goOrigin, 0x01, seed("x0ne interop go log")))
	if err != nil {
		die("log signer: %v", err)
	}
	witSigner, err := fnote.NewSignerForCosignatureV1(skey(goWitness, 0x04, seed("x0ne interop go witness")))
	if err != nil {
		die("witness signer: %v", err)
	}
	var s store
	ref := []byte(`{"interop": "go", "index": 300}` + "\n")
	write(dir, "reference-300.json", ref)
	for i := int64(0); i < nEntries; i++ {
		data := []byte(fmt.Sprintf("x0ne interop entry %d\n", i))
		if i == refIndex {
			data = entry(ref)
		}
		hs, err := tlog.StoredHashes(i, data, &s)
		if err != nil {
			die("%v", err)
		}
		s = append(s, hs...)
		if i+1 == firstSize {
			root, err := tlog.TreeHash(firstSize, &s)
			if err != nil {
				die("%v", err)
			}
			cp := checkpointNote(firstSize, root, logSigner, witSigner)
			write(dir, "checkpoint-301", cp)
			p, err := tlog.ProveRecord(firstSize, refIndex, &s)
			if err != nil {
				die("%v", err)
			}
			write(dir, "reference-300.json.tlog-proof",
				[]byte(fmt.Sprintf("%s\nindex %d\n%s\n%s", proofHead, refIndex, b64lines(p), cp)))
			writeTiles(filepath.Join(dir, "tlog"), &s, firstSize)
		}
	}
	root, err := tlog.TreeHash(nEntries, &s)
	if err != nil {
		die("%v", err)
	}
	write(dir, "tlog/checkpoint", checkpointNote(nEntries, root, logSigner, witSigner))
	writeTiles(filepath.Join(dir, "tlog"), &s, nEntries)
	tp, err := tlog.ProveTree(nEntries, firstSize, &s)
	if err != nil {
		die("%v", err)
	}
	write(dir, "consistency-301-600", []byte(b64lines(tp)))
	write(dir, "vkeys", []byte(fmt.Sprintf("log %s\nwitness %s\n",
		vkey(goOrigin, 0x01, seed("x0ne interop go log")), vkey(goWitness, 0x04, seed("x0ne interop go witness")))))
	fmt.Printf("made: %d entries, checkpoints 301 and 600, tiles, proofs\n", nEntries)
}

// ---- verify a Rust-made TEST log

func parseProof(b []byte) (uint64, [][]byte, []byte) {
	head, cp, ok := bytes.Cut(b, []byte("\n\n"))
	if !ok {
		die("proof without a blank line")
	}
	lines := strings.Split(string(head), "\n")
	if lines[0] != proofHead {
		die("proof header %q", lines[0])
	}
	i := 1
	if strings.HasPrefix(lines[i], "extra ") {
		i++
	}
	idx, err := strconv.ParseUint(strings.TrimPrefix(lines[i], "index "), 10, 64)
	if err != nil || !strings.HasPrefix(lines[i], "index ") {
		die("index line %q", lines[i])
	}
	var hs [][]byte
	for _, l := range lines[i+1:] {
		h, err := base64.StdEncoding.DecodeString(l)
		if err != nil || len(h) != 32 {
			die("hash line %q", l)
		}
		hs = append(hs, h)
	}
	return idx, hs, cp
}

func readKeys(dir string) (string, string, string) {
	b, err := os.ReadFile(filepath.Join(dir, "vkeys"))
	if err != nil {
		die("%v", err)
	}
	m := map[string]string{}
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		k, v, _ := strings.Cut(l, " ")
		m[k] = v
	}
	origin, _, _ := strings.Cut(m["log"], "+")
	return origin, m["log"], m["witness"]
}

func verifyRust(dir string) {
	origin, lk, wk := readKeys(dir)
	logV, err := note.NewVerifier(lk)
	if err != nil {
		die("log vkey %q: %v", lk, err)
	}
	witV, err := fnote.NewVerifierForCosignatureV1(wk)
	if err != nil {
		die("witness vkey %q: %v", wk, err)
	}
	proofs, _ := filepath.Glob(filepath.Join(dir, "*.tlog-proof"))
	sort.Strings(proofs)
	if len(proofs) == 0 {
		die("no proofs in %s", dir)
	}
	for _, p := range proofs {
		b, err := os.ReadFile(p)
		if err != nil {
			die("%v", err)
		}
		ref, err := os.ReadFile(strings.TrimSuffix(p, ".tlog-proof"))
		if err != nil {
			die("%v", err)
		}
		idx, hs, cpNote := parseProof(b)
		cp, _, n, err := flog.ParseCheckpoint(cpNote, origin, logV, witV)
		if err != nil {
			die("%s: checkpoint: %v", p, err)
		}
		cosigned := false
		for _, s := range n.Sigs {
			if s.Name == witV.Name() && s.Hash == witV.KeyHash() {
				cosigned = true
			}
		}
		if !cosigned {
			die("%s: no cosignature verified", p)
		}
		leaf := rfc6962.DefaultHasher.HashLeaf(entry(ref))
		if err := proof.VerifyInclusion(rfc6962.DefaultHasher, idx, cp.Size, leaf, hs, cp.Hash); err != nil {
			die("%s: inclusion: %v", p, err)
		}
	}
	// The tiles of the Rust log against its checkpoint, by compact ranges.
	cpb, err := os.ReadFile(filepath.Join(dir, "tlog", "checkpoint"))
	if err != nil {
		die("%v", err)
	}
	cp, _, _, err := flog.ParseCheckpoint(cpb, origin, logV, witV)
	if err != nil {
		die("tlog/checkpoint: %v", err)
	}
	rf := compact.RangeFactory{Hash: rfc6962.DefaultHasher.HashChildren}
	all := rf.NewEmptyRange(0)
	var blocks [][]byte
	for t := uint64(0); t*tileWidth < cp.Size; t++ {
		w := cp.Size - t*tileWidth
		if w > tileWidth {
			w = tileWidth
		}
		tb, err := os.ReadFile(filepath.Join(dir, "tlog", tilePath(0, t, int(w))))
		if err != nil {
			die("level 0 tile %d: %v", t, err)
		}
		if uint64(len(tb)) != 32*w {
			die("level 0 tile %d: %d bytes", t, len(tb))
		}
		block := rf.NewEmptyRange(t * tileWidth)
		for k := uint64(0); k < w; k++ {
			if err := all.Append(tb[32*k:32*k+32], nil); err != nil {
				die("%v", err)
			}
			if err := block.Append(tb[32*k:32*k+32], nil); err != nil {
				die("%v", err)
			}
		}
		if w == tileWidth {
			r, err := block.GetRootHash(nil)
			if err != nil {
				die("%v", err)
			}
			blocks = append(blocks, r)
		}
	}
	root, err := all.GetRootHash(nil)
	if err != nil {
		die("%v", err)
	}
	if !bytes.Equal(root, cp.Hash) {
		die("root of the level 0 tiles %x, checkpoint %x", root, cp.Hash)
	}
	if len(blocks) > 0 {
		tb, err := os.ReadFile(filepath.Join(dir, "tlog", tilePath(1, 0, len(blocks))))
		if err != nil {
			die("level 1 tile: %v", err)
		}
		if !bytes.Equal(tb, bytes.Join(blocks, nil)) {
			die("level 1 tile differs from the roots of the full level 0 tiles")
		}
	}
	fmt.Printf("verified: %d Rust proofs (log signature, witness cosignature, inclusion), tiles of size %d, %d full blocks\n",
		len(proofs), cp.Size, len(blocks))
}

func main() {
	if len(os.Args) != 3 {
		die("usage: interop make|verify <dir>")
	}
	switch os.Args[1] {
	case "make":
		makeLog(os.Args[2])
	case "verify":
		verifyRust(os.Args[2])
	default:
		die("unknown mode %q", os.Args[1])
	}
}
