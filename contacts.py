from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrieNode:
	children: Dict[str, "TrieNode"] = field(default_factory=dict)
	subtree_count: int = 0

	def insert(self, word: str) -> None:
		current: TrieNode = self
		for ch in word:
			if ch not in current.children:
				current.children[ch] = TrieNode()
			current = current.children[ch]
			current.subtree_count += 1

	def count_prefix(self, prefix: str) -> int:
		current: TrieNode = self
		for ch in prefix:
			if ch not in current.children:
				return 0
			current = current.children[ch]
		return current.subtree_count


def contacts(queries: List[str]) -> List[int]:
	"""
	Perform add/find operations on a contact list using a Trie.
	Each query is of the form 'add name' or 'find partial'.
	Returns a list with the counts for each 'find' query.
	"""
	root = TrieNode()
	results: List[int] = []

	for q in queries:
		parts = q.strip().split()
		if not parts:
			continue
		op = parts[0]
		if op == "add":
			name = parts[1]
			root.insert(name)
		elif op == "find":
			prefix = parts[1]
			results.append(root.count_prefix(prefix))
		else:
			# Ignore unknown operations to be robust
			continue
	return results


if __name__ == "__main__":
	import sys

	data = sys.stdin.read().strip().splitlines()
	if not data:
		sys.exit(0)
	try:
		n = int(data[0].strip())
	except ValueError:
		# If the first line isn't an int, treat all lines as queries
		n = len(data)
		query_lines = data
	else:
		query_lines = data[1:1 + n]

	output = contacts(query_lines)
	for count in output:
		print(count)





