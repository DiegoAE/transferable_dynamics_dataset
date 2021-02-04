""" Violin plot for the SinePD data."""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from DL.evaluation.evaluation import get_evaluation_errors
from DL.utils.plot_utils import get_diego_index, path_to_error_file

matplotlib.rcParams.update({'errorbar.capsize': 3})


class Errors(object):
    def __init__(self, experiment_name='sine_pd'):
        self.history_lengths = [1, 10]
        self.prediction_horizons = [1, 10, 100, 1000]

        self.test_dataset_names = ['training_data',
                                   'validation_data',
                                   'iid_test_data',
                                   'transfer_test_data_1',
                                   'transfer_test_data_2',
                                   'transfer_test_data_3']

        self.experiment_name = experiment_name
        self.errors = {}

    def get_errors(self,
                   method_name,
                   prediction_horizon,
                   history_length,
                   test_dataset_name,
                   error_type):
        key = (method_name,
               prediction_horizon,
               history_length,
               test_dataset_name)

        if not key in self.errors:
            path = path_to_error_file(
                method_name=method_name,
                experiment_name=self.experiment_name,
                prediction_horizon=prediction_horizon,
                history_length=history_length)

            errors = np.load(path)
            self.errors[key] = self.get_evaluation_errors(
                errors[test_dataset_name])

        return self.errors[key][error_type]

    @staticmethod
    def get_evaluation_errors(all_errors):
        evaluation_errors = {}
        evaluation_errors['angle'] = all_errors[:, :, :3]
        evaluation_errors['velocity'] = all_errors[:, :, 3:6]
        evaluation_errors['torque'] = all_errors[:, :, 6:9]

        for key in evaluation_errors.keys():
            norms = np.linalg.norm(evaluation_errors[key], axis=-1, ord=1)
            norms = np.sum(norms, axis=1)

            norms = norms / evaluation_errors[key].size * norms.size
            evaluation_errors[key] = norms.flatten()

        return evaluation_errors

    def create_average_rank_plot(self,
                                 method_names,
                                 prediction_horizons,
                                 test_dataset_names,
                                 history_lengths=None,
                                 error_types=('angle', 'velocity'),
                                 ordered_by=None):

        def get_data(test_dataset_name, prediction_horizon, error_type):
            return self.generate_plotting_data(
                method_names=method_names,
                prediction_horizons=[prediction_horizon],
                history_lengths=history_lengths,
                test_dataset_names=[test_dataset_name],
                error_type=error_type)

        average_ranking = {}
        labels = {}

        n_settings = len(prediction_horizons) * len(error_types)

        for prediction_horizon in prediction_horizons:
            for error_type in error_types:
                for test_dataset_name in test_dataset_names:
                    means, _, labels[test_dataset_name], _ = \
                        get_data(test_dataset_name=test_dataset_name,
                                 prediction_horizon=prediction_horizon,
                                 error_type=error_type)

                    if not test_dataset_name in average_ranking:
                        average_ranking[test_dataset_name] = \
                            np.zeros(means.shape)

                    order = np.argsort(means)
                    ranking = np.empty(means.size)
                    ranking[order] = np.arange(means.size)

                    average_ranking[test_dataset_name] += ranking.astype(
                        float)/n_settings

        if ordered_by is None:
            permutation = None
        else:
            permutation = np.argsort(average_ranking[ordered_by])

        fig, ax = plt.subplots(1, 1)  # , sharey=True, figsize=(20, 4))
        plt.tight_layout()

        plt.gcf().subplots_adjust(top=0.95)
        plt.gcf().subplots_adjust(bottom=0.23)

        plt.gcf().subplots_adjust(left=0.1)

        for test_dataset_name in test_dataset_names:
            graph = average_ranking[test_dataset_name]
            label = labels[test_dataset_name]
            if permutation is not None:
                graph = graph[permutation]
                label = label[permutation]

            ax.plot(np.arange(len(graph)), graph, 'o')

            ax.set_xticks([y for y in range(len(graph))])
            ax.set_xticklabels(label, rotation=90, fontsize=8)
            # self.create_errorbar_plot(means=graph,
            #                           labels=label,
            #
            #                           ax=ax)

        title = 'average ranking'
        if history_lengths is not None and len(history_lengths) == 1:
            title += ', history: ' + str(history_lengths[0])

        ax.set_title(title)

        ax.legend(test_dataset_names)

        return fig, ax

    def create_paper_plot(self,
                          method_names,
                          prediction_horizon,
                          history_lengths=None,
                          test_dataset_names=('iid_test_data',
                                              'transfer_test_data_3'),
                          error_type='angle',
                          ordered_by=None):

        def get_data(test_dataset_name):
            return self.generate_plotting_data(
                method_names=method_names,
                prediction_horizons=[prediction_horizon],
                history_lengths=history_lengths,
                test_dataset_names=[test_dataset_name],
                error_type=error_type)

        if ordered_by is None:
            permutation = None
        else:
            means, _, _, _ = get_data(ordered_by)
            permutation = np.argsort(means)

        fig, ax = plt.subplots(1, 1)  # , sharey=True, figsize=(20, 4))
        plt.tight_layout()

        plt.gcf().subplots_adjust(top=0.95)
        plt.gcf().subplots_adjust(bottom=0.23)

        plt.gcf().subplots_adjust(left=0.1)

        labels = None
        for test_dataset_name in test_dataset_names:
            means, std_devs, local_labels, _ = get_data(test_dataset_name)

            if permutation is not None:
                means = means[permutation]
                std_devs = std_devs[permutation]
                local_labels = local_labels[permutation]

            if labels is None:
                labels = local_labels

            assert ((labels == local_labels).all())

            self.create_errorbar_plot(means=means,
                                      std_devs=std_devs,
                                      labels=labels,
                                      ax=ax)
        title = error_type + ' error  (horizon: ' + str(prediction_horizon)
        if history_lengths is not None and len(history_lengths) == 1:
            title += ', history: ' + str(history_lengths[0])

        title += ')'
        ax.set_title(title)

        ax.legend(test_dataset_names)

        return fig, ax

    def generate_plotting_data(self,
                               method_names,
                               prediction_horizons=None,
                               history_lengths=None,
                               test_dataset_names=None,
                               error_type='angle'):
        # parsing arguments ----------------------------------------------------
        if prediction_horizons is None:
            prediction_horizons = self.prediction_horizons
        if history_lengths is None:
            history_lengths = self.history_lengths
        if test_dataset_names is None:
            test_dataset_names = self.test_dataset_names

        # find suggested description -------------------------------------------
        description = error_type + 'error: '
        description += method_names[0] if len(method_names) == 1 else ''
        description += '   horizon: ' + str(prediction_horizons[0]) if \
            len(prediction_horizons) == 1 else ''
        description += '   history: ' + str(history_lengths[0]) if \
            len(history_lengths) == 1 else ''
        description += '   dataset: ' + test_dataset_names[0] if \
            len(test_dataset_names) == 1 else ''

        # creats lists for labels and errors -----------------------------------
        labels = []
        errors = []
        for method_name in method_names:
            for prediction_horizon in prediction_horizons:
                for history_length in history_lengths:
                    for test_dataset_name in test_dataset_names:
                        label = ''
                        label += method_name if len(method_names) > 1 else ''
                        label += '-pre' + str(prediction_horizon) \
                            if len(prediction_horizons) > 1 else ''
                        label += '-his' + str(history_length) \
                            if len(history_lengths) > 1 else ''
                        label += '---' + test_dataset_name if \
                            len(test_dataset_names) > 1 else ''
                        try:
                            error = self.get_errors(
                                method_name=method_name,
                                prediction_horizon=prediction_horizon,
                                history_length=history_length,
                                test_dataset_name=test_dataset_name,
                                error_type=error_type)
                            labels += [label]
                            errors += [error]
                        except FileNotFoundError:
                            print("Error file for {} not found.".format(label))
        means = np.array(list(map(np.mean, errors)))
        # means = np.mean(errors, axis=1)
        std_devs = np.array(list(map(np.std, errors)))
        # std_devs = np.std(errors, axis=1)
        return means, std_devs, np.array(labels), description

    def create_errorbar_plot(self, means, std_devs, labels, title=None,
                             ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)  # , sharey=True, figsize=(20, 4))

        ax.errorbar(np.arange(len(means)), means,
                    yerr=std_devs,
                    fmt='o')

        ax.set_xticks([y for y in range(len(means))])
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        if not title is None:
            ax.set_title(title)

        # plt.tight_layout()

        return ax


if __name__ == '__main__':

    # Change to 'sym' to plot results over simulated data.
    errors = Errors('sine_pd')

    prediction_horizons = [1, 10, 100]
    history_lengths = [1, 10]
    error_types = ['angle', 'velocity', 'torque']

    # The name of the baseline that predicts no change is 'delta 0'.
    method_names = ['linear', 'avg-linear', 'NN', 'avg-NN',
                    'avg-EQL', 'svgpr', 'avg-svgpr', 'falkon', 'avg-falkon',
                    'system_id_ls', 'system_id_cad']

    path_to_plots = '/tmp'

    # ordered_by_iid_error = [True, False]
    test_dataset_names = ['iid_test_data', 'transfer_test_data_3']
    for ordered_by in test_dataset_names:

        fig, ax = errors.create_average_rank_plot(method_names=method_names,
                                                  prediction_horizons=prediction_horizons,
                                                  history_lengths=history_lengths,
                                                  ordered_by=ordered_by,
                                                  test_dataset_names=test_dataset_names)

        filename = 'average_ranking' + '__ordered_' + ordered_by

        fig.savefig(os.path.join(path_to_plots, filename + '.pdf'),
                    format='pdf')
        fig.savefig(os.path.join(path_to_plots, filename + '.png'))

    for prediction_horizon in prediction_horizons:
        for error_type in error_types:
            for ordered_by_iid_error in [True, False]:
                if ordered_by_iid_error:
                    ordered_by = 'iid_test_data'
                else:
                    ordered_by = None

                fig, ax = errors.create_paper_plot(
                    method_names=method_names,
                    prediction_horizon=prediction_horizon,
                    history_lengths=history_lengths,
                    error_type=error_type,
                    test_dataset_names=test_dataset_names,
                    ordered_by=ordered_by)

                filename = error_type + '__horizon_' + \
                    str(prediction_horizon).zfill(4)

                if ordered_by_iid_error:
                    filename += '__ordered_iiderror'
                else:
                    filename += '__ordered_method'

                fig.savefig(os.path.join(path_to_plots, filename + '.pdf'),
                            format='pdf')
                fig.savefig(os.path.join(path_to_plots, filename + '.png'))

