build:
	python setup.py build

sdist:
	python setup.py sdist

bdist:
	python setup.py bdist

clean:
	rm -rf dist/

publish:
	twine upload dist/* -r pypi
