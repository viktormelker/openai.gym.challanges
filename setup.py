from setuptools import find_packages, setup

setup(
    name="openai_solver",
    version="0.1.0",
    description="Openai Gym solver functions",
    url="https://github.com/viktormelker/openai.gym.challanges",
    author="Viktor Melker",
    author_email="",
    license="MIT",
    packages=find_packages(),
    install_requires=["keras>=2.2.4", "h5py>=2.9.0"],
    zip_safe=False,
)
