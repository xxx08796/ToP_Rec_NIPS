from matplotlib.dates import MicrosecondLocator
import pandas as pd
from collections import defaultdict
from ToP.llm import LLMMessage, LLMRegistry
import random
from ToP.utils import safe_parser, async_task_runner
import json
import asyncio
import yaml

class TreeNode:
    def __init__(self, name, criteria=None, id=None, tweets= None):
        self.name = name
        self.criteria = criteria
        self.sub_branches:list["TreeNode"] = []
        self.parent:"TreeNode" = None
        self.tweets:list[int] = tweets if tweets is not None else []

    def add_sub_branch(self, sub_node:"TreeNode"):
        self.sub_branches.append(sub_node)
        sub_node.parent = self

    def remove_sub_branch(self, sub_node:"TreeNode"):
        assert sub_node in self.sub_branches, f"sub_node not in sub_branches, sub_node: {sub_node.name}, parent: {self.name}, sub_branches: {[sb.name for sb in self.sub_branches]}"
        self.sub_branches.remove(sub_node)
        sub_node.parent = None
    
    def add_tweet(self, tweet_id:int):
        self.tweets.append(tweet_id)
    
    def is_leaf(self):
        return not self.sub_branches 

    def is_root(self):
        return self.parent is None

    def get_tweets(self) -> list[int]:
        '''
        get the tweet ids loaded in this node
        if the node is not leaf, then return all the tweets in its sub branches
        '''
        if self.is_leaf():
            return self.tweets
        else:
            return [tweet for sub_branch in self.sub_branches for tweet in sub_branch.get_tweets()]
    
    def get_path(self):
        if self.is_root():
            return self.name
        else:
            return self.parent.get_path() + "->" + self.name
        
    def get_load(self):
        '''
        return the number of tweets loaded on the tree
        '''
        if self.get_tweets():
            return len(self.get_tweets())
        else:
            return 0
        #return len(self.get_tweets())
    
    def get_leaves(self) -> list["TreeNode"]:

        if self.is_leaf():
            return [self]
        else:
            return [leaf for sub_branch in self.sub_branches for leaf in sub_branch.get_leaves()]
    
    def find_ancestors(self) -> list["TreeNode"]:
        ancestors = []
        cur = self
        while cur:
            ancestors.append(cur)
            cur = cur.parennt
        return ancestors


    def to_dict(self):
        """
        output the dictionary format for save
        """
        pass

    def pretty_print(self, show_criteria=True, show_load=True) -> str:
        """
        Tree visualization
        """
        lines = []
        connectors = []
        
        def _build_tree_lines(node: TreeNode, depth: int, is_last: bool) -> None:
            prefix = "".join(connectors[:-1])
            prefix += "└── " if is_last else "├── "
            
            # Node Description
            node_info = node.name
            if show_criteria and node.criteria:
                node_info += f" ({node.criteria})"
            if show_load:
                node_info += f" [Load: {node.get_load()}]"
            if node.is_leaf():
                node_info += " 🍂"
            
            lines.append(prefix + node_info)
            
            # 处理子节点
            child_count = len(node.sub_branches)
            for i, child in enumerate(node.sub_branches):
                # 更新连接符号
                if depth > 0:
                    connectors.append("    " if connectors[-1] == "└── " else "│   ")
                else:
                    connectors.append("    " if i == child_count-1 else "│   ")
                
                _build_tree_lines(child, depth + 1, i == child_count - 1)
                
                connectors.pop()

        lines.append(self.name + " 🌱" + (" [Root]" if self.is_root() else ""))
        for i, child in enumerate(self.sub_branches):
            connectors.append("    " if i == len(self.sub_branches)-1 else "│   ")
            _build_tree_lines(child, 0, i == len(self.sub_branches)-1)
            connectors.pop()
            
        return "\n".join(lines)

        


class TreeManager:
    def __init__(self, tree: TreeNode, tweet_df:pd.DataFrame, model_name: str, prompts=None):
        self.tree = tree  
        self.tweet_df = tweet_df  
        self.model = LLMRegistry().get(model_name)
        self.prompts = prompts
        self.max_try = 3
        self.sample_size = 100 # for large leafs
    
    def get_tweets(self, id_list:list[int], sample_size = None, return_df:bool = False) -> list[str]:
        '''
        retrieve tweets from the tweet_df with given id_list
        return:
            return_df==False: list of str tweet contents
            return_df==True:  sub_df
        '''
        sub_df = self.tweet_df[self.tweet_df['mid'].isin(id_list)]
        if sample_size:
            sub_df = sub_df.sample(n=sample_size)
        if return_df:
            return sub_df
        else:
            return sub_df['content'].tolist()

    def prune_empty_nodes(self, node:TreeNode):
        parent_node = node.parent
        if node.get_load() == 0:
            parent_node.remove_sub_branch(node)
       
    

    async def merge_nodes(self, node1:TreeNode, node2:TreeNode):
        assert node1.is_leaf() and node2.is_leaf(), "Only leaf nodes can be merged"
        assert node1.parent == node2.parent, "Nodes to be merged must have the same parent"
        parent_node = node1.parent
        print(f"combining the nodes {node1.name} and {node2.name}, parent node {parent_node.name}")

        if len(parent_node.sub_branches) == 2:
            parent_node.tweets = parent_node.get_tweets() 
            parent_node.remove_sub_branch(node1)
            parent_node.remove_sub_branch(node2)
        else:
            #TODO write the prompt here for generate a new node name 
            #parent node has more than two leafs
            path1 = node1.get_path()
            path2 = node2.get_path()
            mids = node1.get_tweets()+ node2.get_tweets()
            tweets = self.get_tweets(mids)
            leaf_combine_prompt = self.prompts["leaf_combine"].format(
                path_1=path1,
                path_2=path2,
                tweets = tweets
            )
            msg = LLMMessage(role="user", content=leaf_combine_prompt)
            for _ in range(self.max_try):
                try:
                    response = await self.model.generate_response_async(messages=[msg])
                    response = safe_parser(response)
                    assert isinstance(response, dict), f"Expected response to be a dictionary, but got {type(response)}"
                    node_name = response["name"]
                    combine_node = TreeNode(
                        name = node_name,
                        tweets = mids,
                    )
                    parent_node.add_sub_branch(combine_node)
                    parent_node.remove_sub_branch(node1)
                    parent_node.remove_sub_branch(node2)
                    return
                except Exception as e:
                    print(e)
                    print("retrying")



    async def split_node(self, node:TreeNode):
        """
        add two subbranches for a overloaded leaf node
        """
        if not node.is_leaf():
            raise Exception("Cannot split a non-leaf node")
        print(f"Spliting node: {node.name}")
        mids = node.get_tweets().copy()
        path = node.get_path()
        if len(mids) > self.sample_size:
            sampled_mids = random.sample(mids, self.sample_size)
            sampled_tweets = self.get_tweets(sampled_mids)
            leaf_split_prompt = self.prompts["leaf_split"].format(
                path = path,
                tweets = sampled_tweets
            )
        else:
            leaf_split_prompt = self.prompts["leaf_split"].format(
                path = path,
                tweets = mids
            )
        msg = LLMMessage(role="user", content=leaf_split_prompt)
        ### TODO
        for _ in range(self.max_try):
            try:
                response = await self.model.generate_response_async(messages=[msg])
                response = safe_parser(response)
                name1, name2 = response["name1"], response["name2"]
                node1, node2 = TreeNode(name = name1), TreeNode(name = name2)
                print(f"Split into two nodes {node1.name} and {node2.name}")
                node.add_sub_branch(node1)
                node.add_sub_branch(node2)
                node.tweets = []
                break
            except Exception as e:
                print(e)
                print("retrying")
        await self.tweets_reassign(node1=node1, node2=node2, mids=mids)


    
    
    async def tweets_reassign(self, node1:TreeNode, node2:TreeNode, mids: list[int]):
        """
        reassign the tweets to the split new nodes
        """
        idx_map = {
            1:node1.name,
            2:node2.name
        }
        async def tweet_reassign(tweet_content:str, mid:int) -> None:
            """
            reassign the tweet to the new nodes
            return the path index
            """
            reassign_prompt = self.prompts["tweet_reassign"].format(
                path_1 = node1.get_path(),
                path_2 = node2.get_path(),
                tweet = tweet_content,
                idx_map = idx_map
            )
            msg = LLMMessage(role="user", content=reassign_prompt)
            for _ in range(self.max_try):
                try:
                    response = await self.model.generate_response_async(messages=[msg])
                    #print(response)
                    response = safe_parser(response)
                    path_idx = int(response["path_idx"]) if isinstance(response["path_idx"], str) else response["path_idx"]
                    name = response["name"]
                    assert isinstance(path_idx, int), f"wrong path_idx type {type(path_idx)}, path_idx: {path_idx}, response: {response}"
                    assert path_idx in [1,2], f"wrong path_idx {path_idx}"

                    if name == idx_map[path_idx]:
                        if path_idx == 1:
                            node1.tweets.append(mid)
                            return
                        else:
                            node2.tweets.append(mid)
                            return
                    else:
                        raise ValueError(f"wrong leaf name {name}, actual leaf name {idx_map[path_idx]}")
                except Exception as e:
                    print(e)
                    print("retrying")
                

        tasks = []
        tweet_df = self.get_tweets(id_list=mids, return_df=True)
        for mid in mids:
            row = tweet_df[tweet_df["mid"] == mid]
            content = row["content"]
            tasks.append(tweet_reassign(content, mid = mid))
        await async_task_runner(tasks, describe="reassigning tweets")
    
    async def rebalance_tree(self, 
                        split_ratio: float = 1.5, 
                        merge_ratio: float = 0.3,
                        max_operations: int = 20):
        """
        Dynamically rebalance the tree
        """
        node = self.tree
        print(node.pretty_print())
        

        all_leaves = node.get_leaves()
        avg_load = sum(leaf.get_load() for leaf in all_leaves) / len(all_leaves) if all_leaves else 0
        split_threshold = avg_load * split_ratio
        merge_threshold = avg_load * merge_ratio
        
        print(f"Current Average Load: {avg_load:.1f} | Split Threshold: {split_threshold:.1f} | Combine Threshold: {merge_threshold:.1f}")  

        operation_count = 0
        processed = set()
        
        queue = [node]
        
        while queue and operation_count < max_operations:
            current = queue.pop(0)

            if current.is_leaf() and current not in processed:
                load = current.get_load()
                
                    
                if load < merge_threshold:
                    parent = current.parent
                    if parent:

                        siblings = [sib for sib in parent.sub_branches 
                                if sib != current and sib.is_leaf() and sib not in processed]
                        
                        if siblings:
                            best_candidate = min(siblings, key=lambda x: x.get_load())
                            
                            await self.merge_nodes(current, best_candidate)
                            processed.update([current, best_candidate])
                            operation_count += 1
                            continue
      
                elif load > split_threshold:
                    await self.split_node(current)
                    processed.add(current)
                    operation_count += 1
                    continue
                    
            # add sub branches to the queue
            queue.extend(current.sub_branches)
        
        # prune all empty nodes

        all_leaves = node.get_leaves()
        empty_leaves = [leaf for leaf in all_leaves if leaf.get_load() == 0]
        for leaf in empty_leaves:
            self.prune_empty_nodes(leaf)

        node = self.tree
        print(node.pretty_print())
        all_leaves = node.get_leaves()
        
        avg_load = sum(leaf.get_load() for leaf in all_leaves) / len(all_leaves) if all_leaves else 0
        
        split_threshold = avg_load * split_ratio
        merge_threshold = avg_load * merge_ratio
        
        print(f"Current Average Load: {avg_load:.1f} | Split Threshold: {split_threshold:.1f} | Combine Threshold: {merge_threshold:.1f}")  
        

    def tree_to_dict(self):
        """
        output the rebalanced tree as the dictionry with the reassinged tweets
        """
        root = self.tree
        current_id = [1]  
        tweet_df = self.tweet_df.copy()
    
        def _traverse(node: TreeNode) -> dict:
            node_dict = {"name": node.name}
            
            # 保留criteria字段（如果有）
            if node.criteria is not None:
                node_dict["criteria"] = node.criteria
                
            # 处理叶子节点
            if node.is_leaf():
                node_dict["id"] = current_id[0]
                mids = node.tweets
                tweet_df.loc[tweet_df["mid"].isin(mids), "leaf_id"] = current_id[0]

                current_id[0] += 1  # ID自增
            # 处理内部节点
            else:
                node_dict["sub_branches"] = []
                for child in node.sub_branches:
                    node_dict["sub_branches"].append(_traverse(child)) 
            return node_dict
        tree_dict = _traverse(root)
        tree_dict["root"] = root.name
        return tree_dict, tweet_df



################################################
#Helper functions for the formart transformation
################################################


def dict_to_tree(tree_dict: dict, df: pd.DataFrame) -> TreeNode:
    """
    Convert a nested dictionary structure to a TreeNode tree.
    
    Args:
        tree_dict: The nested dictionary representing the tree structure
        df: DataFrame containing tweet IDs indexed by leaf node IDs
        
    Returns:
        The root TreeNode of the constructed tree
    """
    # Create root node
    root = TreeNode(
        name=tree_dict["root"],
        criteria=tree_dict.get("criteria")
    )
    
    # Recursively build the tree
    if "sub_branches" in tree_dict:
        for sub_dict in tree_dict["sub_branches"]:
            sub_node = _build_subtree(sub_dict, df)
            root.add_sub_branch(sub_node)
    
    return root

def _build_subtree(node_dict: dict, df: pd.DataFrame) -> TreeNode:
    """
    Helper function to recursively build a subtree from a dictionary node.
    """
    # Create the current node
    node = TreeNode(
        name=node_dict["name"],
        criteria=node_dict.get("criteria", None)
    )
    
    if "sub_branches" in node_dict:
        # This is an internal node - process its children
        for sub_dict in node_dict["sub_branches"]:
            sub_node = _build_subtree(sub_dict, df)
            node.add_sub_branch(sub_node)
    else:
        # This is a leaf node - get tweets from dataframe
        leaf_id = node_dict["id"]
        tweets = df[df["leaf_id"] == leaf_id]["mid"].to_list()
        if tweets:
            node.tweets = tweets
        else:
            node.tweets = []
    return node

