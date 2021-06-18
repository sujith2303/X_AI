import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="X_AI",
    version="0.0.1",
    author="Sujith Anumala",
    author_email="sujithanumala23@gmail.com",
    description="Easy way to build your custom AI projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sujith2303/X_AI",
    project_urls={
        "Bug Tracker": "https://github.com/sujith2303/X_AI/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)