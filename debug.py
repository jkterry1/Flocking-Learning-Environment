import numpy as np
values = np.array([-30, 5, None, 10, 50, -20, -1, 15, None, 17, -3, 200, -100])
scratch_values = [-np.inf if i is None else i for i in values]
ordered_indices = np.argsort(scratch_values)[::-1]
print([values[i] for i in ordered_indices])


# for i in range(len(values)):
#     index = scratch_values.index(max(scratch_values))
#     ordered_indices.append(index)
#     scratch_values[index] = -201

# print()
# for i in ordered_indices:
#     print(values[i])

#print(ordered_indices)
