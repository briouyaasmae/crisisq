# Upload this repository

This is a public author-attributed release. Extract the ZIP and publish the CONTENTS
of its `CrisisQ/` folder at one repository root. Do not upload the ZIP as a single
file or mix private author/submission packages into the repository.

For a new empty GitHub remote, use GitHub Desktop or run these commands from the
extracted folder after replacing the URL:

```sh
git init
git add .
git commit -m "Release CrisisQ research code and evidence"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The commands assume a new empty remote. `.cache/`, `.reproduced/`, local models,
downloaded posts, environment folders and build products are ignored. Exclude them
from browser uploads too. `CITATION.cff` contains the author list. Add a real repository
URL and accepted DOI only after they exist.

MIT covers supplied application/analysis code. Models, data, manuscript rights and
the Elsevier class have separate terms. Preserve upstream and third-party notices.
