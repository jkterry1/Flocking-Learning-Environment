from setuptools import find_packages, setup


setup(
    name='fle',
    version=0.1,
    author='Swarm Labs',
    author_email="justinkterry@gmail.com",
    description="Flocking Learning Environment",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "gym"],
    python_requires=">=3.6, <3.10",
    packages=["pettingzoo"] + ["pettingzoo." + pkg for pkg in find_packages("pettingzoo")],
    include_package_data=True,
    install_requires=[
        "pettingzoo",
        "pybind11"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
