import cnnscore
import pytest


class TestTranspose(object):

	def test_none_data(self):
		data = None
		with pytest.raises(TypeError):
			cnnscore.transpose(data)

	def test_empty_data(self):
		data = []
		ret = cnnscore.transpose(data)
		assert ret == []

	def test_singleton(self):
		data = [0]
		with pytest.raises(TypeError):
			cnnscore.transpose(data)

	def test_inner_empty(self):
		data = [[]]
		ret = cnnscore.transpose(data)
		assert ret == []

	def test_inner_singleton(self):
		data = [[1]]
		ret = cnnscore.transpose(data)
		assert ret == [[1]]

	def test_one_by_two(self):
		data = [[1, 2]]
		ret = cnnscore.transpose(data)
		assert ret == [[1], [2]]

	def test_two_by_one(self):
		data = [[1], [2]]
		ret = cnnscore.transpose(data)
		assert ret == [[1, 2]]

	def test_two_by_two(self):
		data = [[1, 2], [3, 4]]
		ret = cnnscore.transpose(data)
		assert ret == [[1, 3], [2, 4]]

	def test_ragged(data):
		data = [[1, 2], [3]]
		ret = cnnscore.transpose(data)
		assert ret == [[1, 3]]

class TestSortByIndex(object):

	def test_none_data(self):
		data, index = None, 0
		with pytest.raises(TypeError):
			cnnscore.sort_by_index(data, index)

	def test_none_index(self):
		data, index = [[1], [3], [2]], None
		with pytest.raises(TypeError):
			cnnscore.sort_by_index(data, index)

	def test_shallow_data(self):
		data, index = [1, 3, 2], 0
		with pytest.raises(TypeError):
			cnnscore.sort_by_index(data, index)

	def test_out_of_order(self):
		data, index = [[1], [3], [2]], 0
		ret = cnnscore.sort_by_index(data, index)
		assert ret == [[1], [2], [3]]

	def test_in_order(self):
		data, index = [[1], [2], [3]], 0
		ret = cnnscore.sort_by_index(data, index)
		assert ret == [[1], [2], [3]]

	def test_reverse_order(self):
		data, index = [[1], [2], [3]], 0
		ret = cnnscore.sort_by_index(data, index, reverse=True)
		assert ret == [[3], [2], [1]]

	def test_ragged_data(self):
		data, index = [[0, 2], [0], [0, 1]], 1
		with pytest.raises(IndexError):
			cnnscore.sort_by_index(data, index)

class TestApplyToFields(object):

	def test_none_data(self):
		data, funcs = None, [int]
		with pytest.raises(TypeError):
			cnnscore.apply_to_fields(data, funcs)

	def test_none_funcs(self):
		data, funcs = [['0', 0], ['1', 1]], None
		with pytest.raises(TypeError):
			cnnscore.apply_to_fields(data, funcs)

	def test_short_func(self):
		data, funcs = [['0', 0], ['1', 1]], [int]
		with pytest.raises(TypeError):
			cnnscore.apply_to_fields(data, funcs)

	def test_diff_return(self):
		data, funcs = [['0', 0], ['1', 1]], [int, str]
		ret = cnnscore.apply_to_fields(data, funcs)
		assert ret == [[0, '0'], [1, '1']]

	def test_same_return(self):
		data, funcs = [['0', 0], ['1', 1]], [str, int]
		ret = cnnscore.apply_to_fields(data, funcs)
		assert ret == [['0', 0], ['1', 1]]

	def test_ragged_data(self):
		data, funcs = [['0', 0], ['1']], [int, str]
		ret = cnnscore.apply_to_fields(data, funcs)
		assert ret == [[0, '0'], [1, 'None']]

class TestGroupByIndex(object):

	def test_none_data(self):
		data, index = None, 0
		with pytest.raises(TypeError):
			cnnscore.group_by_index(data, index)

	def test_none_index(self):
		data, index = [[1, 1], [1, 2], [2, 3], [3, 4]], None
		with pytest.raises(TypeError):
			cnnscore.group_by_index(data, index)

	def test_shallow_data(self):
		data, index = [1, 3, 2], 0
		with pytest.raises(TypeError):
			cnnscore.group_by_index(data, index)

	def test_only_groups(self):
		data, index = [[1], [3], [2]], 0
		ret = cnnscore.group_by_index(data, index)
		assert ret == {1:[[]], 2:[[]], 3:[[]]}

	def test_one_group(self):
		data, index = [[1, 1], [1, 2], [1, 3], [1, 4]], 0
		ret = cnnscore.group_by_index(data, index)
		assert ret == {1:[[1], [2], [3], [4]]}

	def test_two_groups(self):
		data, index = [[1, 1], [2, 2], [1, 3], [2, 4]], 0
		ret = cnnscore.group_by_index(data, index)
		assert ret == {1:[[1], [3]], 2:[[2], [4]]}

