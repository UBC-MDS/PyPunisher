# Docs

Guide used: [sphinx-for-python](https://gisellezeno.com/tutorials/sphinx-for-python-documentation.html).

Note: this guide recommends you target a `docs` directory outside the root. Conversely, here we target a directory 
within the root (making version control easier).

## Updating the docs

```
$ cd docs
$ make html
$ cd build/html
$ git checkout gh-pages  # should be on this branch already, but just to sure.
$ git add .
$ git commit -m "SOME COMMIT MESSAGE HERE"
$ git push origin gh-pages
```
