"""
This code is adapted from the GitHub repository at: https://github.com/valentinlageard/melodendron.
"""
from vocabulary import end_token_index
import random
random.seed(42)


class VOMMNode:
    """
    Class for a tree node in a Variable-Order Markov Model.
    """

    def __init__(self, value, continuation):
        """
        Params:
        - int value: the node value
        - int continuation: the following event
        """
        # Each node has a value
        self.value = value
        # Each node has a set of continuations
        self.continuations = {continuation}
        # Each node has a set of children
        self.children = set()

    def __getitem__(self, key):
        """
        Get the child node with value equal to the key when node[key] is called.
        Params:
        - int key: the child node value
        Return:
        - VOMMNode child (optional): the child node with value = key
        """
        for child in self.children:
            if child.value == key:
                return child
        return None

    def add_child(self, child):
        """
        Add a child node to the node's set of children.
        Params:
        - VOMMNode child: the new child node
        """
        self.children.add(child)

    def add_continuation(self, continuation):
        """
        Add a continuation to the node's set of continuations.
        Params:
        - int continuation: the new continuation
        """
        self.continuations.add(continuation)


class VOMM:
    """
    Class for a Variable-Order Markov Model designed for fast retrieval using a tree structure.
    """

    def __init__(self, max_order=8):
        """
        Params:
        - int max_order (default = 8): the maximum number of previous events that can be used to decide the next event
        """
        # A set of all roots for the trees in the model
        self.roots = set()
        # A list of every event sequence used to train the model
        self.training_event_sequences = []
        # A list of every training event
        self.training_events = []
        # The maximum order of the model
        self.max_order = max_order

    def insert(self, continuation, context_values):
        """
        Insert a new continuation into the model trees with context.
        Params:
        - int continuation: an event in the training sequence
        - list[int] context_values: the previous (up to) `max_order` events in the training sequence
        """
        # If no context, cannot insert into tree
        if len(context_values) == 0:
            print("No context given, so cannot insert into tree.")
            return

        # Get last context value
        last_context_val = context_values[-1]

        root_found = False
        for root in self.roots:
            # If the last context value is that of a tree root
            if root.value == last_context_val:
                # Get root node
                current_node = root
                # Add the new continuation to that node
                current_node.add_continuation(continuation)
                # Indicate there is a root for last context value
                root_found = True
                break

        # If the last context value is not that of a tree root
        if root_found == False:
            # Create a new tree with this as the root
            current_node = VOMMNode(last_context_val, continuation)
            self.roots.add(current_node)

        # Iterate through rest of the context values in reverse order (right to left)
        for context_value in context_values[-2::-1]:
            next_node = current_node[context_value]
            # If there doesn't exist a child of the current node with value equal to the context value
            if next_node is None:
                # Create a node with this value
                next_node = VOMMNode(context_value, continuation)
                # Add the node as a child of the current node
                current_node.add_child(next_node)
            # If there does
            else:
                # Add the continuation to this child node
                next_node.add_continuation(continuation)
            # Move down the tree to the child node
            current_node = next_node

    def insert_sequence(self, event_sequence):
        """
        Insert a sequence of events into the model.
        Params:
        - list[int] event_sequence: the training event sequence
        """
        self.training_event_sequences.append(event_sequence)
        self.training_events += event_sequence

        # For each event in the sequence
        for i, event in enumerate(event_sequence):
            if i == 0:
                # Don't insert first event since it has no context
                pass
            elif i < self.max_order:
                # Insert the event into the model trees using the previous events as context
                self.insert(event, event_sequence[:i])
            else:
                # Insert the event into the model trees using the previous `max_order` events as context
                self.insert(event, event_sequence[(i - self.max_order): i])

    def get_continuations(self, context_values, mode="regular", c=None):
        """
        Get the possible continuations given a set of context values and count how often the continuations appear 
        while traversing.
        Params:
        - list[int] context_values: the previous events in the current sequence
        - str mode (default = "regular"): the mode determining how to traverse the tree to get the continuations
        ("regular": traverse as far as possible using the context,
         "c_min": traverse only nodes with at least c continuations using the context)
        - int c (optional): must be a positive integer if mode = "c_min"
        Return:
        - set continuations: all possible continuations (next events) given these previous events
        - dict continuation_counts: a count of how many times each continuation appeared while traversing the tree
        """
        # If no context, cannot get continuations
        if not context_values:
            print("No context given, so cannot get continuations.")
            return None, None

        # Check that c is given if mode is "c_min"
        if mode == "c_min" and c is None:
            print('Mode is "c_min" but c not given.')
            return None, None

        # Get last context value
        last_context_val = context_values[-1]

        # Count the number of times each continuation occurs as we traverse the tree
        continuation_counts = {}

        for root in self.roots:
            # If the last context value is that of a tree root
            if root.value == last_context_val:
                # Get root node
                current_node = root
                # Update the continuation counts
                for i in current_node.continuations:
                    if i not in continuation_counts:
                        continuation_counts[i] = 1
                    else:
                        continuation_counts[i] += 1
                # Iterate through rest of the context values in reverse order (right to left)
                for context_value in context_values[-2::-1]:
                    # Traverse tree to previous context value
                    next_node = current_node[context_value]
                    # If there doesn't exist a child of the current node with value equal to the context value
                    if next_node is None:
                        # Stop traversing tree and return the current node's continuations
                        return current_node.continuations, continuation_counts
                    else:
                        # If we want to traverse to the point where there are at least c continuations
                        if mode == "c_min":
                            # If the child node has less than c continuations
                            if len(next_node.continuations) < c:
                                # Stop traversing tree and return the current node's continuations
                                return current_node.continuations, continuation_counts
                        # Traverse the tree to the child node
                        current_node = next_node
                        # Update the continuation counts
                        for i in current_node.continuations:
                            if i not in continuation_counts:
                                continuation_counts[i] = 1
                            else:
                                continuation_counts[i] += 1
                return current_node.continuations, continuation_counts

        # If no tree root has this value, then no part of the context sequence has been seen before, so return None
        return None, None

    def next_event(self, context_events, selector="context_length_weight", mode="regular", c=None):
        """
        Returns the next event given a series of context events given a selector.
        Params:
        - list[int] context_values: the previous events in the current sequence
        - str selector (default = "context_length_weight"): the approach to selecting the next event
        ("random": select randomly from possibilities,
         "event_count_weight": select using how often the events occur in the corpus as weights,
         "context_length_weight": select using the context lengths of the continuations as weights)
        - str mode (default = "regular"): the mode determining how to traverse the tree to get the continuations
        ("regular": traverse as far as possible using the context,
         "c_min": traverse only nodes with at least c continuations using the context)
        - int c (optional): must be a positive integer if mode = "c"
        Return:
        - int next_event: the chosen next event
        """
        # Get the possible events given the context events
        continuations, continuation_counts = self.get_continuations(context_events, mode, c)

        if continuations is not None:
            if selector == "context_length_weight":
                # Randomly choose one of the events that appeared while traversing the tree with the counts as weights
                next_event = random.choices(list(continuation_counts.keys()), list(continuation_counts.values()))[0]
            elif selector == "event_count_weight":
                # Get counts for how many times the possible next events appeared in the training sequences
                counts = []
                for cont_idx in continuations:
                    counts.append(self.training_events.count(cont_idx))
                next_event = random.choices(list(continuations), weights=counts)[0]
            else:
                # Otherwise, choose next event randomly
                next_event = random.choices(list(continuations))[0]
        else:
            # If no part of the context has been seen before, choose a random next event from the training events
            next_event = random.choice(self.training_events)

        return next_event

    def generate_sequence(self, start=[end_token_index], selector="context_length_weight", mode="regular", c=None):
        """
        Generate a sequence until an end token is added or a keyboard interrupt occurs.
        Params:
        - list[int] start (default = [end_token_index]): the token(s) with which to start generating from
        - str selector (default = "context_length_weight"): the approach to selecting the next event
        ("random": select randomly from possibilities,
         "event_count_weight": select using how often the events occur in the corpus as weights,
         "context_length_weight": select using the context lengths of the continuations as weights)
        - str mode (default = "regular"): the mode determining how to traverse the tree to get the continuations
        ("regular": traverse as far as possible using the context,
         "c_min": traverse only nodes with at least c continuations using the context)
        - int c (optional): must be a positive integer if mode = "c"
        """
        # Initialise first element of sequence to some random event from training sequences
        while start == [end_token_index]:
            start = random.choices(self.training_events)
        sequence = start

        try:
            i = 0
            while True:
                if i < self.max_order:
                    # Get next event based on the previous events
                    next_event = self.next_event(sequence, selector, mode, c)
                else:
                    # Get next event based on the previous `max_order` events
                    next_event = self.next_event(sequence[-self.max_order:], selector, mode, c)

                # If end token, end generation
                if next_event == end_token_index:
                    return sequence

                sequence.append(next_event)

                i += 1

        except KeyboardInterrupt:
            return sequence
