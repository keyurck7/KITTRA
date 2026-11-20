from setuptools import setup

package_name = "kittra_bridge"

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
    description="MuJoCo <-> ROS 2 bridge for the Kittra kitting cell",
    license="MIT",
    entry_points={"console_scripts": ["mujoco_bridge = kittra_bridge.mujoco_bridge:main"]},
)
