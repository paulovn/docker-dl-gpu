
NAME := dl-utils

VERSION_FILE := VERSION.txt
VERSION := $(shell tr -d '\n ' < $(VERSION_FILE))
PKG := dist/$(NAME)-$(VERSION).tar.gz


# -----------------------------------------------------------------------

all: $(PKG)


install: all
	pip install --upgrade $(PKG)

clean:
	rm -f $(PKG)

uninstall:
	pip uninstall $(NAME)


# -----------------------------------------------------------------------

$(PKG):  $(VERSION_FILE)
	python setup.py sdist

