from setuptools import setup, find_packages


setup(
    name="gvhmr",
    version="1.0.0",
    packages=find_packages(),
    author="Zehong Shen",
    description="GVHMR training, inference, and web tooling",
    url="https://github.com/zju3dv/GVHMR",
    entry_points={
        "console_scripts": [
            "gvhmr-web=hmr4d.service.server:main",
            "gvhmr-sync-assets=hmr4d.service.assets:main",
        ]
    },
)
