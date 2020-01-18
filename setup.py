from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="practical",
    version="0",
    description="work in progress",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
    ],
    url="https://github.com/mythologic/practical",
    author="Max Humber",
    author_email="max.humber@gmail.com",
    license="MIT",
    py_modules=["practicaltrees"],
    python_requires=">=3.6",
    setup_requires=["setuptools>=38.6.0"],
)
