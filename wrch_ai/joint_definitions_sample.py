# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import wrnchAI

definitions = wrnchAI.JointDefinitionRegistry.available_definitions()
for definition in definitions:
    wrnchAI.JointDefinitionRegistry.get(definition).print_joint_definition()
