build:
	python setup.py build

test:
	tox

sdist:
	python setup.py sdist

bdist:
	python setup.py bdist

clean:
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -rf build/

publish:
	twine upload dist/* -r pypi
