from vertical_profile import VerticalProfile

################################################
# TEST


test_samples = {1.0: (1.0, 1.0), 2.0: (2, 2), 3.0: (3, 3), 4.0: (4, 4), 5.0: (5, 5), 6.0: (6, 6), 7.0: (7, 7), 8.0: (8, 8), 9.0: (9, 9)}
sd = VerticalProfile(test_samples, None)

levels = [1, 3, 5, 7, 9]


result1 = sd.rescale(levels)

print(result1)