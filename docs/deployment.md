# Deployment

This repository publishes all of `zkunkworks.com` with GitHub Pages.

The deployment artifact has two layers:

- `website/` is copied to the artifact root and serves `https://zkunkworks.com/`.
- MkDocs is built into `public/zynamics` and serves `https://zkunkworks.com/zynamics/`.

## GitHub Settings

In `jonathanhuml/zynamics`, configure:

1. Go to `Settings` -> `Pages`.
2. Set `Build and deployment` -> `Source` to `GitHub Actions`.
3. Set `Custom domain` to `zkunkworks.com`.
4. After the certificate is issued, enable `Enforce HTTPS`.

When publishing with a custom GitHub Actions workflow, a checked-in `CNAME` file
is not required.

## DNS Records

At the DNS provider for `zkunkworks.com`, point the apex domain at GitHub Pages:

| Type | Name | Value |
| --- | --- | --- |
| `A` | `@` | `185.199.108.153` |
| `A` | `@` | `185.199.109.153` |
| `A` | `@` | `185.199.110.153` |
| `A` | `@` | `185.199.111.153` |

Optionally add IPv6 records:

| Type | Name | Value |
| --- | --- | --- |
| `AAAA` | `@` | `2606:50c0:8000::153` |
| `AAAA` | `@` | `2606:50c0:8001::153` |
| `AAAA` | `@` | `2606:50c0:8002::153` |
| `AAAA` | `@` | `2606:50c0:8003::153` |

For `www.zkunkworks.com`, add:

| Type | Name | Value |
| --- | --- | --- |
| `CNAME` | `www` | `jonathanhuml.github.io` |

Do not use wildcard DNS records for this domain.

## Verify

DNS changes can take up to 24 hours to propagate. Check the apex records with:

```bash
dig zkunkworks.com +noall +answer -t A
```

After the workflow deploys:

```text
https://zkunkworks.com/
https://zkunkworks.com/zynamics/
```
