#!/bin/bash
jupyter-nbconvert --to slides lecture_01.ipynb --reveal-prefix=reveal.js --post serve
