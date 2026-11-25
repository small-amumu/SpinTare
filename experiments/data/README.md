# Texts

| File | Size | Content | Used by |
|---|---|---|---|
| `long_text.txt` | 214 KB | LongBench passage-retrieval contexts ("Paragraph N: ...") | probes (`kvprobe_unified.py`), calibration |
| `qasper_text.txt` | 302 KB | LongBench Qasper contexts | LC1 test + calibration (each calibrates the other) |
| `mfqa_text.txt` | 316 KB | LongBench MultiFieldQA-en contexts | LC1 |
| `musique_text.txt` | 362 KB | LongBench MuSiQue contexts, md5 2db28f75 | exemption-gate simulation (C23) windows |
| `diag_long_text.txt` | 277 KB | LongBench multi-hop QA contexts, longer cut, md5 4604ba1d | C23 windows |
| `diag_text.txt` | 63 KB | LongBench multi-hop QA contexts ("Passage N: ...") | engine KL check, mu calibration |
| `pg19_text.txt` | 705 KB | PG19 (Project Gutenberg, public domain) book text, md5 be2f3aee | E2 llama.cpp length study up to 32K |
| `pg19_text2.txt` | 498 KB | a second PG19 text, md5 e8e76e78; `pg19_text + pg19_text2` = `pg19_long` (md5 0a55cfc4) | E2 at 64K / 128K |

Derived from THUDM/LongBench (concatenated contexts, no questions or answers); used only as calibration /
evaluation text. Scripts read them from `$KVREL_DATA`; link `work/data -> data` or set `KVREL_DATA=$PWD/data`.
