# Sources

Raw HTML was saved manually in a browser on 2026-11-14 (UTC), because the site rejects
non-browser clients. Raw files live in `corpus/raw/` (gitignored). `cdc.md` and
`decreto_7962.md` are generated from them with
`uv run python -m support_agent.corpus build-laws`.

| doc_id | URL | downloaded (UTC) | SHA-256 of raw HTML |
|---|---|---|---|
| cdc | https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm | 2026-11-14 | ad435530d39af7da4114b42ca1695f6af9376bc72bdff80273a235cd3a0bbfc5 |
| decreto_7962 | https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2013/decreto/d7962.htm | 2026-11-14 | 9e419d5bdb9e3c69e1522d05a3c8777e30873f5ed38cc45e708313060cdf542a |
