# Personal Website

Link to my [personal website](https://srecharan.github.io/).

---

## ⚠️ Maintenance Mode (Site is currently OFFLINE)

The website is **temporarily disabled**. Visitors see a blank "Site Unavailable" page.

### How to bring the site back online

1. Open `.github/workflows/deploy.yml`
2. Find this line near the top (line ~67):
   ```yaml
   MAINTENANCE_MODE: 'true'
   ```
3. Change it to:
   ```yaml
   MAINTENANCE_MODE: 'false'
   ```
4. Commit and push. The GitHub Actions workflow will rebuild and deploy the full site automatically.

That's it — one word change, and your entire website is back exactly as it was. No other files were modified.

---