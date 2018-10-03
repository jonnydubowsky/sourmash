
# Releasing a new version of sourmash


These are adapted from the khmer release docs, originally written by
Michael Crusoe.

Remember to update release numbers/RC in:

* this document
* sourmash/VERSION

## Testing a release


 1\. The below should be done in a clean checkout:
```
cd $(mktemp -d)
git clone git@github.com:dib-lab/sourmash.git
cd sourmash
```
2\. Set your new version number and release candidate:
```
        new_version=1.0
        rc=rc1
```
 and then tag the release candidate with the new version number prefixed by
   the letter 'v':
```
        git tag -a v${new_version}-${rc}
        git push --tags git@github.com:dib-lab/sourmash.git
```
3\. Test the release candidate. Bonus: repeat on Mac OS X:
```
        cd ..
        virtualenv testenv1
        virtualenv testenv2
        virtualenv testenv3
        virtualenv testenv4
        # First we test the tag

        cd testenv1
        source bin/activate
        git clone --depth 1 --branch v${new_version}-${rc} https://github.com/dib-lab/sourmash.git
        cd sourmash
        pip install -r requirements.txt
        make test

        # Secondly we test via pip

        cd ../../testenv2
        deactivate
        source bin/activate
        pip install -U setuptools
        pip install -e git+https://github.com/dib-lab/sourmash.git@v${new_version}-${rc}#egg=sourmash[test]
        cd src/sourmash
        make test
        make dist
        cp dist/sourmash*tar.gz ../../../testenv3/

        # Is the distribution in testenv2 complete enough to build another
        # functional distribution?

        cd ../testenv3/
        deactivate
        source bin/activate
        pip install -U setuptools
        pip install sourmash*tar.gz
        pip install pytest
        tar xzf sourmash*tar.gz
        cd sourmash*
        pip install -r requirements.txt
        make dist
        make test
```
4\. Publish the new release on the testing PyPI server.  You will need
   to change your PyPI credentials as documented here:
   https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives
   We will be using `twine` to upload the package to TestPyPI and verify
   everything works before sending it to PyPI:
```
        pip install twine
        twine upload --repository-url https://test.pypi.org/legacy/ sourmash*.tar.gz
```
   Test the PyPI release in a new virtualenv:
```
        cd ../../testenv4
        deactivate
        source bin/activate
        pip install -U setuptools
        # install as much as possible from non-test server!
        pip install screed pytest numpy matplotlib scipy khmer ijson
        pip install -i https://test.pypi.org/simple --pre sourmash
```
5\. Do any final testing:

   * check that the binder demo notebook is up to date

## How to make a final release

When you've got a thoroughly tested release candidate, cut a release like
so:

1\.Create the final tag and publish the new release on PyPI (requires an
   authorized account):
```
        cd ../sourmash
        git tag v${new_version}
        make dist
        twine upload dist/*.tar.gz
```
2\. Delete the release candidate tag and push the tag updates to GitHub:
```
        git tag -d v${new_version}-${rc}
        git push git@github.com:dib-lab/sourmash.git
        git push --tags git@github.com:dib-lab/sourmash.git
```
3\. Add the release on GitHub, using the tag you just pushed.  Name
   it 'version X.Y.Z', and copy and paste in the release notes:

## Bioconda

Open a new PR in Bioconda. This is the file that needs to be changed:
https://github.com/bioconda/bioconda-recipes/blob/master/recipes/sourmash/meta.yaml

Usually you need to change the `version` variable in the first line
and the `sha256` under the `source` section.

This is an example PR (for `2.0.0a10`): https://github.com/bioconda/bioconda-recipes/pull/11197

## To test on a blank Ubuntu system


```

   apt-cache update && apt-get -y install python-dev libfreetype6-dev && \
   pip install sourmash[test]
```
