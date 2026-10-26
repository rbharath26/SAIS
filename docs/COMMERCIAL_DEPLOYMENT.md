# Commercial on-premise deployment

SAIS is licensed as proprietary software under `EULA.md`. It is designed to
run locally with a customer-managed Ollama server; no SAIS-hosted server is
required.

## License options

Sell one annual option per order:

- **Named user:** one identified person may use SAIS on their controlled
  devices.
- **Designated machine:** SAIS runs on one identified physical or virtual
  machine.

Record the customer, license type, quantity, start/end date, support tier, and
license-key identifier in an order form or invoice. Do not distribute the
source repository publicly under an open-source license.

## Delivery

1. Build the wheel: `py -m pip wheel --no-deps . --wheel-dir dist`
2. Deliver the wheel and `EULA.md` through an authenticated customer channel.
3. Issue a signed license file or key after the commercial agreement is
   accepted. License-key verification is not yet implemented in this package.
4. Customer installs Ollama and separately obtains an embedding model such as
   `qwen3-embedding:0.6b`, subject to the model's own terms.
5. Provide `THIRD_PARTY_NOTICES.md` if any third-party dependency or binary is
   bundled with the customer delivery.

## Before first sale

- Have counsel review `EULA.md` and complete governing-law, contact, privacy,
  tax, export-control, and invoicing terms.
- Verify every third-party dependency, model, and benchmark license.
- Generate a release-specific software bill of materials and preserve the
  corresponding notices.
- Add license-key validation if technical enforcement is required.
- Maintain private release artifacts and a customer-license register.
