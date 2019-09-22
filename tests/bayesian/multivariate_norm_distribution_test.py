import unittest
import numpy as np

from statistics.bayesian.multivariate_norm_distribution import MultivariateNormalDistribution


class TestStringMethods(unittest.TestCase):

    def test_multivariate_normal_distribution(self):
        dim = 5
        # 教師データ
        mu_teacher = np.random.uniform(-10, 10, (dim,))
        x_vector = np.random.uniform(-3, 3, (dim, 100))
        x_vector = x_vector - np.mean(x_vector)
        sigma_teacher = np.dot(x_vector, x_vector.T)
        num_sampling = 200000
        x_data = np.random.multivariate_normal(mu_teacher, sigma_teacher, num_sampling)  # 教師データ

        # モデルセット
        mnd = MultivariateNormalDistribution(dim)

        # 学習
        for i in range(num_sampling):
            mnd.update(x_data[i])

        # 推論(生成)
        sample = mnd.get_sample()
        y = mnd.pdf(np.random.uniform(-10, 10, (dim,)))
        print('\nsampling: \n{}'.format(sample))
        print('\npdf: \n{}'.format(y))

        # 学習されているかテスト
        mu_pred, sigma_pred = mnd.get_parameters()
        print('\nteacher mu: \n{}'.format(mu_teacher))
        print('\npred mu: \n{}'.format(mu_pred))
        print('\nteacher sigma: \n{}'.format(sigma_teacher))
        print('\npred sigma: \n{}'.format(sigma_pred))
        if int(np.abs(np.sum(mu_teacher - mu_pred))) != 0:
            raise ValueError(
                'Did not train the mu. the difference is {}'.format(np.abs(np.sum(mu_teacher - mu_pred))))
        elif int(np.abs(np.sum(sigma_teacher - sigma_pred))) > 5 * dim:
            raise ValueError(
                'Did not train the sigma. the difference is {}'.format(np.abs(np.sum(sigma_teacher - sigma_pred))))


if __name__ == '__main__':
    unittest.main()
