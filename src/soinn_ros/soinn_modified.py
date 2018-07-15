import numpy as np
import soinn

class Soinn_modified(object):
    """
    Modified version of Self-Organizing Incremental Neural
    Network (SOINN).
    Modified in order to create a new SOINN model from a 
    previous one, designed to take as input the pair
    [previous_signal, new_feature], the new feature is also
    being assigned a new metrics different from the previous
    one.
    """

    def __init__(self, previous_soinn):
        """
        :param previous_soinn: The original SOINN model that this new version will adapt.
        :return:
        """
        self.delete_node_period = previous_soinn.delete_node_period
        self.max_edge_age = previous_soinn.max_edge_age
        self.min_degree = previous_soinn.min_degree
        self.num_signal = previous_soinn.num_signal
        self.nodes = previous_soinn.nodes
        self.winning_times = previous_soinn.winning_times
        self.adjacent_mat = previous_soinn.adjacent_mat

        # Add one dimension to the nodes and signals and set the last components of existing nodes to -1
        self.dim = previous_soinn.dim + 1
        n = self.node.shape[0]
        self.nodes.resize((n, self.dim))
        self.nodes[:, -1] = np.full((n, 1), -1)
    
    def input_signal(self, signal, learning=True):
        """ Input a new signal to SOINN
        :param signal: A new input signal
        :return:
        """
        self.__check_signal(signal)
        self.num_signal += 1

        if self.nodes.shape[0] < 3:
            self.__add_node(signal)
            return

        winner, dists = self.__find_nearest_nodes(2, signal)

        if not learning:
_
_
_rity_thresholds(winner)
_sts[1] > sim_thresholds[1]:
_
_
_
_1])
_s(winner[1])
_gnal)
_r[1], signal)
_
        if self.num_signal % self.delete_node_period == 0:
            self.__delete_noise_nodes()
        return winner

    def __check_signal(self, signal):
        """ check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as self.dim.
        So, this method have to be called before calling functions that use self.dim.
        :param signal: an input signal
        """
        if not(isinstance(signal, np.ndarray)):
            raise TypeError()
        if len(signal.shape) != 1:
            raise TypeError()
        if not(hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                raise TypeError()

    def __add_node(self, signal):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim))
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))

    def __find_nearest_nodes(self, num, signal, mahar=True):
        """
        Modified: add a bonus/malus if the last component of the signal is the same/different from the compared node.
        Neutral if compared_node[-1] == -1
        """
        #if mahar: return self.__find_nearest_nodes_by_mahar(num, signal)
        n = self.nodes.shape[0]
        indexes = [0.0] * num
        sq_dists = [0.0] * num
        D = util.modified_calc_distance(self.nodes, np.asarray([signal] * n))
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    def __find_nearest_nodes_by_mahar(self, num, signal):
        indexes, sq_dists = util.calc_mahalanobis(self.nodes, signal, 2)
        return indexes, sq_dists

    def calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0:
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:
                pal_indexes = []
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = util.modified_calc_distance(self.nodes[pal_indexes], np.asarray([self.nodes[i]] * len(pal_indexes)))
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    def __add_edge(self, node_indexes):
        self.__set_edge_weight(node_indexes, 1)

    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    def __delete_old_edges(self, winner_index):
        candidates = []
        for k, v in self.adjacent_mat[winner_index, :].items():
            if v > self.max_edge_age + 1:
                candidates.append(k[1])
                self.__set_edge_weight((winner_index, k[1]), 0)
        delete_indexes = []
        for i in candidates:
            if len(self.adjacent_mat[i, :]) == 0:
                delete_indexes.append(i)
        self.__delete_nodes(delete_indexes)
        delete_count = sum([1 if i < winner_index else 0 for i in delete_indexes])
        return winner_index - delete_count

    def __set_edge_weight(self, index, weight):
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    def __update_winner(self, winner_index, signal):
        self.winning_times[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w)/self.winning_times[winner_index]

    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w)/(100 * self.winning_times[i])

    def __delete_nodes(self, indexes):
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        #_old_ver_adjacent_mat = self.adjacent_mat[np.ix_(remained_indexes, remained_indexes)]
        self.__update_adjacent_mat(indexes, n, len(remained_indexes))
        #assert (_old_ver_adjacent_mat.toarray() == self.adjacent_mat.toarray()).all()

    def __update_adjacent_mat(self, indexes, prev_n, next_n):
        while indexes:
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    continue
                if key1 > indexes[0]:
                    new_key1 = key1 - 1
                else:
                    new_key1 = key1
                if key2 > indexes[0]:
                    new_key2 = key2 - 1
                else:
                    new_key2 = key2
                #dok_matrix.__getitem__ is slow.
                #So access as dictionary
                next_adjacent_mat[new_key1, new_key2] = super(dok_matrix, self.adjacent_mat).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i-1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    def __delete_nodes2(self, indexes):
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        self.adjacent_mat = self.adjacent_mat[np.ix_(remained_indexes, remained_indexes)]

    def __delete_noise_nodes(self):
        n = len(self.winning_times)
        noise_indexes = []
        for i in range(n):
            if len(self.adjacent_mat[i, :]) < self.min_degree:
                noise_indexes.append(i)
        if noise_indexes:
            self.__delete_nodes(noise_indexes)

    def print_info(self):
        print('Total Nodes: {0}'.format(len(self.nodes)))

    def save(self, dumpfile='soinn.dump'):
        import joblib
        joblib.dump(self, dumpfile, compress=True, protocol=0)