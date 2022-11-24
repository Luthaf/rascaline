# -*- coding: utf-8 -*-
import numpy as np
from equistore import Labels, TensorBlock, TensorMap


class Transformer:
    """The 'Transformer' class makes it easy to create a representation matrix
    when using some other matrix as a reference. A classic use case is to create
    a TensorMap representation for a dataset, then perform transformations
    within that TensorMap (e.g., keys_to_features or keys_to_properties), and
    select the most useful features in the transformed TensorMap.
    The 'Transformer' allows a set of these features to be used to calculate
    a new TensorMap, thus saving computation time and maintaining a single
    representation for all representations.

    Parameters:
    -----------
    :param selectors: The selector to be used when selecting the features.
        Currently the same selector is used for different blocks.
        #TODO: It should be possible for the user to pass a list of selectors
        for blocks.
    :param transformation: the type of transformation to be performed.
        Two options are possible - 'keys_to_features' and 'keys_to_properties'.
        #TODO: provide the ability to pass a list of conversions that will occur
        one after the other.
    :param moved_keys: Those keys which will be moved during the transformation.
        This variable can accept knowledge of type str (one key), list (a list
        of keys) and Labels (in addition to the name of the keys, pass a list
        of keys to be moved).
    """

    def __init__(self, selector, transformation=None, moved_keys=None):
        #
        self.selector = selector
        self.transformation = transformation
        self.moved_keys = moved_keys
        if (
            (self.transformation is not None)
            and (self.transformation != "keys_to_samples")
            and (self.transformation != "keys_to_properties")
        ):
            raise ValueError(
                "`transformation` parameter should be either `keys_to_samples`,"
                f" either `keys_to_properties`, got {self.transformation}"
            )
        if (self.transformation is None) and (self.moved_keys is not None):
            raise ValueError("unable to shift keys: unknown transformation type")

    def _copy(self, tensor_map):
        """This function that allows you to create a copy of 'TensorMap'.
        It may be worth adding this function to the equistore.
        """
        blocks = []
        for _, block in tensor_map:
            blocks.append(block.copy())
        return TensorMap(tensor_map.keys, blocks)

    def keys_definition(self):
        """This is another internal function that performs two main tasks.
        First, it converts all moved_keys to the same format.  What is
        meant is that further we need the names of the keys we are going
        to move, as well as the 'TensorMap' keys, which will be passed
        to the compute function as a reference at the end. This function
        stores the names of the moved keys in the 'moved_keys_names' array,
        and stores the keys of the final TensorMap reference in 'final_keys'.
        """
        # the first 2 cases are simple - we either copy the moved_keys directly,
        # or create an array based on them, and simply take all the keys passed
        # in the fit TensorMap step as the final keys.
        if isinstance(self.moved_keys, str):
            self.moved_keys_names = [self.moved_keys.copy()]
            self.final_keys = self._old_keys
        elif isinstance(self.moved_keys, list):
            self.moved_keys_names = self.moved_keys.copy()
            self.final_keys = self._old_keys
        else:
            # The third case is a little more complicated.
            # First, we save the names of the moved keys,
            # taking them from Labels 'moved_keys'.
            self.moved_keys_names = self.moved_keys.names
            names = []
            new_keys = []
            # Let's write down the order of the keys we will have during the
            # course of the algorithm in the 'names'
            names.extend(self.tensor_map.keys.names)
            names.extend(self.moved_keys_names)
            # Now let's generate reference TensorMap keys. They will consist of
            # two parts - those keys that were left after transformation, and
            # those keys that were in the values of the variable moved_keys.
            # Go through them and create all possible combinations of these
            # parts.
            for key in self.tensor_map.keys:
                for value in self.moved_keys:
                    clue = [k.copy() for k in key]
                    clue.extend(value)
                    new_keys.append(clue)
            # The keys have been listed in random order, let's arrange them and
            # store the values in 'final_keys'.
            indices = []
            for key in self._old_keys_names:
                indices.append(names.index(key))
            ordered_keys = []
            for el in new_keys:
                key = [el[i] for i in indices]
                ordered_keys.append(key)
            self.final_keys = Labels(
                names=self._old_keys_names, values=np.array(ordered_keys)
            )

    def _mover(self, tensor_map):
        # Internal function that does the transformation of the reference
        # Tensormap.
        self._old_keys = tensor_map.keys
        self._old_keys_names = tensor_map.keys.names
        tensor_copy = self._copy(tensor_map)
        if self.transformation is not None:
            if self.transformation == "keys_to_samples":
                tensor_copy.keys_to_samples(self.moved_keys)
            elif self.transformation == "keys_to_properties":
                tensor_copy.keys_to_properties(self.moved_keys)
        return tensor_copy

    def properties_selection(self):
        # This function selects properties according to a preset algorithm
        # within each 'TensorMap' block
        blocks = []
        for _, block in self.tensor_map:
            mask = self.selector.fit(block.values).get_support()
            selected_properties = block.properties[mask]
            blocks.append(
                TensorBlock(
                    # Since the resulting 'TensorMap' will then be used as a
                    # reference, the only thing we are interested in each
                    # block is the name of the properties.
                    values=np.empty((1, len(selected_properties))),
                    samples=Labels.single(),
                    components=[],
                    properties=selected_properties,
                )
            )

        self.selected_tensor = TensorMap(self.tensor_map.keys, blocks)

    def fit(self, tensor_map):
        """The fit function tells the transformer which attributes to use when
        creating new representations.

        Parameters:
        -----------
        :param tensor_map: reference TensorMap, with which transformations are
            carried out and in which properties are selected.
        """
        self.tensor_map = self._mover(tensor_map)
        self.keys_definition()
        self.properties_selection()

    def transform(self, frames, calculator):
        """A function that creates a TensorMap representation based on the
        passed frames as well as a previously performed fit.

        Parameters:
        -----------
        :param frames: list with the frames to be processed during this function.
        :param calculator: calculator that will compute the representation of
            the transferred frames.
        """
        if self.transformation is None:
            # trivial case - nothing happened, do the usual calculation.
            descriptor = calculator.compute(
                frames, selected_properties=self.selected_tensor
            )
            return descriptor
        elif self.transformation == "keys_to_samples":
            # In the second case the situation is a bit more complicated.
            # Suppose we originally had a set of key names {'a', 'b', 'c'}.
            # We moved key 'c' to samples. We are left with blocks with keys
            # {'a', 'b'}. Let's start going through all the final keys. We take
            # key {a_1, b_1, c_1}. Its corresponding features are in the
            # {a_1, b_1} block. Accordingly, all we need to do is tear off what
            # we have moved from the keys, take the properties from the
            # resulting block and save them.
            blocks = []
            idx = []
            # save the positions of the moved keys.
            for key in self.moved_keys_names:
                idx.append(self.final_keys.names.index(key))
            for obj in self.final_keys:
                # separate the moved keys, obtain a block based on the remainder
                obt_key = tuple(item for i, item in enumerate(obj) if i not in idx)
                if len(obt_key) == 0:
                    obt_key = (0,)
                block = self.selected_tensor.block(
                    self.tensor_map.keys.position(obt_key)
                )
                blocks.append(
                    TensorBlock(
                        values=np.empty((1, len(block.properties))),
                        samples=Labels.single(),
                        components=[],
                        properties=block.properties,
                    )
                )
            properties_tensor = TensorMap(self.final_keys, blocks)
            # Do the final computation
            descriptor = calculator.compute(
                frames,
                selected_properties=properties_tensor,
                selected_keys=self.final_keys,
            )
            return descriptor
        elif self.transformation == "keys_to_properties":
            # The third case is the most complicated. Again, let's start with a
            # TensorMap with {'a', 'b', 'c'} keys. Suppose we move the 'c' keys
            # to properties. We take the final key {a_1, b_1, c_1}. Its
            # corresponding properties lie in the block {a_1, b_1}. But we do
            # not need all the properties, we need only those properties that
            # include c_1 in the label. We need to take all these properties,
            # separate c_1 from them and save them in the corresponding block.

            # save positions of the moved keys in the properties array
            pos_in_prop = []
            for key in self.moved_keys_names:
                pos_in_prop.append(self.tensor_map.property_names.index(key))
            idx = []
            property_names = []
            # save property names, which were originaly, before the `move`
            for i, key in enumerate(self.tensor_map.property_names):
                if i not in pos_in_prop:
                    property_names.append(key)
            # determine the positions of the moved keys in the final keys
            for key in self.moved_keys_names:
                idx.append(self.final_keys.names.index(key))
            # in this dictionary we write a list of properties, which we will
            # save for each block.
            properties_dict = {}
            for obj in self.final_keys:
                obj_tuple = tuple(item for item in obj)
                properties_dict[obj_tuple] = []
            # running through all the keys of the transformed tensor
            for obj in self.tensor_map.keys:
                # obtain block by the position of key
                block = self.selected_tensor.block(self.tensor_map.keys.position(obj))
                # go through all properties (each one consists of a set of values)
                for prop in block.properties:
                    # this array stores the part of properties that was previously
                    # keys
                    add_key = []
                    # and here are those who always have been properties
                    property_initial = []
                    for i, item in enumerate(prop):
                        if i in pos_in_prop:
                            add_key.append(item)
                        else:
                            property_initial.append(item)
                    obt_key = []
                    add_key_ind = 0
                    key_ind = 0
                    # put the key together from the two pieces - the one you
                    # moved and the one you have left
                    for i in range(len(self.final_keys.names)):
                        if i in idx:
                            obt_key.append(add_key[add_key_ind])
                            add_key_ind += 1
                        else:
                            obt_key.append(obj[key_ind])
                            key_ind += 1
                    obt_key = tuple(obt_key)
                    # add the original properties in our dictionary
                    properties_dict[obt_key].append(property_initial)
            blocks = []
            # go through the original keys to create a tensor for selection
            for key in self.final_keys:
                key = tuple(key)
                # In theory, we may find that we have not selected any property
                # that is correspond to this block - take this into account.
                if properties_dict[key] != []:
                    values = np.array(properties_dict[key])
                else:
                    values = np.empty((0, len(property_names)), dtype=int)
                properties = Labels(names=property_names, values=values)
                # create the block for each key
                blocks.append(
                    TensorBlock(
                        values=np.empty((1, len(properties))),
                        samples=Labels.single(),
                        components=[],
                        properties=properties,
                    )
                )
            properties_tensor = TensorMap(self.final_keys, blocks)
            descriptor = calculator.compute(
                frames,
                selected_properties=properties_tensor,
                selected_keys=self.final_keys,
            )
            return descriptor
