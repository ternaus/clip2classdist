from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("-f")]

setup(
    name="clip2classdist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,  # Use the parsed requirements here
    entry_points={
        "console_scripts": [
            "clip2classdist=clip2class_dist.compute:main",
        ],
    },
    author="Vladimir Iglovikov",
    author_email="iglovikov@gmail.com",
    description="A Python script that analyzes image classes using OpenAI CLIP model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ternaus/clip2classdist",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
