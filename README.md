# g1moon.github.io

Personal academic website for Jiwon Moon (문지원), built on the same theme/layout as
[hwanheelee1993.github.io](https://hwanheelee1993.github.io/) (orderedlist "Minimal" theme).

## Structure

```
index.html                  # the whole site — edit this
stylesheets/styles.css      # theme stylesheet (copied from the template)
stylesheets/pygment_trac.css
javascripts/scale.fix.js
materials/profile.jpg       # profile photo (800x800)
```

## Things left to fill in

Search `index.html` for `TODO`:

1. **M.S. start date** (line ~78) — the `<sup>` next to the SNU education entry.
2. **B.S. education entry** (~line 87) — the whole block is commented out; fill in the
   university / major / date and remove the comment markers.
3. **Scholarships & Awards** (~line 133) — the list is empty; add `<li>` entries.

Twitter and CV links in the header are intentionally commented out. To enable the CV,
drop the PDF at `materials/CV_Jiwon_Moon.pdf` and uncomment that block.

## Deploy to GitHub Pages

```bash
git init
git add .
git commit -m "Initial site"
git branch -M main
git remote add origin https://github.com/g1moon/g1moon.github.io.git
git push -u origin main
```

The repo **must** be named `g1moon.github.io` for the user-site URL. Then in
Settings → Pages, set Source = "Deploy from a branch", Branch = `main` / `(root)`.
The site goes live at https://g1moon.github.io/ after a minute or two.

## Preview locally

```bash
python3 -m http.server 8000 --directory .
# open http://localhost:8000
```
