from setuptools import find_packages, setup

setup(
    name='fgvc9herbarium',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={
        '': 'src',
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'train=herbariumtools.train:run',
            'validate=herbariumtools.validate:run',
            'predict=herbariumtools.predict:run',
            'explain=herbariumtools.explain:run',
            'swaify=herbariumtools.swaify:run',
            'grid-search=herbariumtools.grid_search:run',
            'prepare-data=herbariumtools.prepare_data:run',
            'find-lr=herbariumtools.find_lr:run',
            'dsviewer=herbariumtools.dsviewer:run',
            'dsinfo=herbariumtools.dsinfo:run',
            'minfo=herbariumtools.model_info:run',
        ]
    }
)

