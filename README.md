# g1moon.github.io

Personal academic website. Live at https://g1moon.github.io/

## Structure

```
index.html                  # the whole site — edit this
stylesheets/styles.css      # theme stylesheet
stylesheets/pygment_trac.css
javascripts/scale.fix.js
materials/profile.jpg       # profile photo (800x800)
```

## Things left to fill in

Search `index.html` for `TODO`:

1. **M.S. start date** — the `<sup>` next to the SNU education entry.
2. **B.S. education entry** — the block is commented out; fill in the university /
   major / date and remove the comment markers.
3. **Scholarships & Awards** — the list is empty; add `<li>` entries.

Twitter and CV links in the header are intentionally commented out. To enable the CV,
drop the PDF at `materials/CV_Jiwon_Moon.pdf` and uncomment that block.

## Publish

Push to `main`; GitHub Pages rebuilds automatically and the site is live in a minute or two.

```bash
git add -A && git commit -m "Update site" && git push
```

## Preview locally

```bash
python3 -m http.server 8000 --directory .
# open http://localhost:8000
```
