from setuptools import setup

package_name = "kittra_policy"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Keyur Chaudhari",
    description="Trained-policy runner for the Kittra cell",
    license="MIT",
    entry_points={"console_scripts": ["policy_runner = kittra_policy.policy_runner:main"]},
)
