# Third-Party Notices

vLLM Manager includes open-source Go and JavaScript dependencies. Their license metadata is recorded in `go.sum` and `frontend/package-lock.json` and is checked by `scripts/check-licenses.sh`.

The dependency set currently contains software under Apache-2.0, MIT, ISC, and BSD-3-Clause licenses. Each dependency retains its own copyright and license terms. Source locations and exact versions are available from the two lock files.

The optional monitoring setup downloads Grafana community dashboards 22515 revision 1 and 25263 revision 1 at setup time. Those dashboard files are not distributed in vLLM Manager packages.
