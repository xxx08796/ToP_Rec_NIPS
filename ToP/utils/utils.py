import json
import re
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from tqdm.asyncio import tqdm_asyncio
import random

def tree_structure(node:dict, level=0):
    """
    convert the dict tree to a string format
    """
    indent = '  ' * level

    if level == 0:
        node_name = node["root"]
    else:
        node_name = node["name"]

    if 'id' in node:
        node_name += f" (leaf_node_id: {node['id']})"
    result = f"{indent}├── {node_name}\n"
    
    if 'sub_branches' in node and node['sub_branches']:
        for child in node['sub_branches']:
            result += tree_structure(child, level + 1)
            
    return result

def validate_tree_structure(node: dict, is_root=True, depth=0, id_counter=None):
    # process the root node
    if id_counter is None:
        id_counter = [1] 

    if is_root:
        if 'root' not in node:
            raise ValueError("Lack root field in the tree structure")

        if 'criteria' not in node:
            raise ValueError(f"Lack criteria in root {current_name} ")
        branches = node.get('sub_branches', [])
        current_name = node['root']
    else:
        if 'name' not in node:
            raise ValueError(f"Find unnamed node at depth {depth}") 
        branches = node.get('sub_branches', [])
        current_name = node['name']
    
    # public logic
    if branches:
        if 'criteria' not in node:
            raise ValueError(f"Lack criteria at non-leaf node {current_name}")
        
        names = []
        for child in branches:
            if 'name' not in child:
                raise ValueError(f"Find unnamed node at {current_name}")
            names.append(child['name'])
        if len(names) != len(set(names)):
            raise ValueError(f"Node {current_name} has duplicated child names: {names}")
    else:
        node['id'] = id_counter[0]
        id_counter[0] += 1
        node.pop('sub_branches', None)
        if 'criteria' in node:
            del node['criteria']                        

    for child in branches:
        validate_tree_structure(child, is_root=False, depth=depth+1, id_counter = id_counter)
    return True  


def find_tree_by_category(
    category: str, 
    category_trees: List[dict]
) -> Tuple[Optional[int], Optional[Dict], Dict[int, str], List[Dict[str, object]]]:
    '''
    params:
        category: the category to find
        category_trees: the list of category trees
        return_paths: whether to return leaf paths 
    return:
        max_id: the max id in the tree (None if not found)
        tree: the tree structure (None if not found)
        leaf_id_to_name: a dict mapping leaf_id to leaf_name
        leaf_paths: list of {"path": "root > ... > leaf", "id": leaf_id}
    '''
    def find_max_index(tree: dict) -> int:
        """Find the max id in the tree (assuming ids are in order)"""
        if "id" in tree:
            return tree["id"]
        if "sub_branches" in tree:
            for sub_node in reversed(tree["sub_branches"]):
                max_id = find_max_index(sub_node)
                if max_id:
                    return max_id
        return None

    def collect_leaves(tree: dict, leaf_map: Dict[int, str]) -> None:
        """Collect leaf node ids and names"""
        if "id" in tree:
            leaf_map[tree["id"]] = tree["name"]
        elif "sub_branches" in tree:
            for sub_node in tree["sub_branches"]:
                collect_leaves(sub_node, leaf_map)

    def extract_leaf_paths(tree: dict, current_path: str = "") -> List[Dict[str, object]]:
        """Extract all leaf paths"""
        paths = []
        node_name = tree.get("name") or tree.get("root", "")
        new_path = f"{current_path} > {node_name}" if current_path else node_name

        if "id" in tree:
            paths.append({"path": new_path, "id": tree["id"]})
        elif "sub_branches" in tree:
            for sub_node in tree["sub_branches"]:
                paths.extend(extract_leaf_paths(sub_node, new_path))
        return paths


    for tree in category_trees:
        if tree and tree.get('root') == category:
            max_id = find_max_index(tree)
            leaf_id_to_name = {}
            collect_leaves(tree, leaf_id_to_name)
            leaf_paths = extract_leaf_paths(tree)
            return max_id, tree, leaf_id_to_name, leaf_paths
    
    return None, None, {}, []

def balance_check(tree:dict):
    """
    check if the tree is split balanced
    the ratio of tweets assigned to each leaves
    """
    pass

def safe_parser(response: str) -> dict:
        '''
        safe parser annoation
        '''
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if match:
                json_str = match.group(1).strip()
            else:
                
                matches = re.findall(r'\{[\s\S]*?\}', response)
                if not matches:
                    raise ValueError("No valid JSON found in response")
                json_str = matches[-1]  

            clean_response = json.loads(json_str)
            return clean_response

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}\nExtracted JSON:\n{json_str}")
        except Exception as e:
            print(f"Parsing failed: {e}\nResponse content:\n{response}")



async def async_task_runner(tasks, max_concurrent=32, describe="Processing tasks"):
    """
    General asynchronous task scheduler that supports task concurrency control and progress bar display.
    :param tasks: A list of tasks to be executed (already created coroutines).
    :param max_concurrent: The maximum number of tasks to run concurrently.
    :return: A list of task results returned in the original order.
    """
    sem = asyncio.Semaphore(max_concurrent)  # 限制最大并发数
    results = []
    
    async def wrapped_task(task):
        async with sem:
            return await task
    
    # 使用 tqdm_asyncio 显示进度条
    results = await tqdm_asyncio.gather(*(wrapped_task(task) for task in tasks), desc=describe)
    
    return results



