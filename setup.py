# usr/bin/env python
# -*- coding: utf-8 -*-
import glob2
# import glob -- for python3.6
from distutils.cmd import Command
from setuptools import setup


class AutoFormatCommand(Command):
    description = "Run autopep8 on all python files in the pwd"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import autopep8
        autopep8_options = {"max_line_length": 100}
        # for filepath in glob.iglob("**/*.py", recursive=True): -- python3.6
        for filepath in glob2.iglob("**/*.py"):  # recursive=True):
            # with open(filepath, "rt" encoding="utf-8") as fp: -- python3.6
            with open(filepath, "rt") as fp:
                before = fp.read()
            after = autopep8.fix_code(before, options=autopep8_options)
            if before != after:
                # with open(filepath, "wt" encoding="utf-8") as fp: -- python3.6
                with open(filepath, "wt") as fp:
                    fp.write(after)
                # print("fModified {filepath}") -- python3.6
                print("Modified {}".format(filepath))


setup(name="multimodal_vrl_camera_net",
      cmdclass={"format": AutoFormatCommand})
