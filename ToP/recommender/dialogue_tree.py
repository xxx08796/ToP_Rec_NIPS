from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ToP.llm import LLMMessage, LLMRegistry

class DialogueTree(BaseModel):
    name:str
    type:str
    dialogue:List[LLMMessage] = []
    sub_branches:List["DialogueTree"] = []

    def __repr__(self):
        pass

    def find_dialogue_path(self, name: str) -> Optional[List[LLMMessage]]:
        """
        Given a node name, return the dialogue history from the root to the node.
        """
        if self.name == name:
            return self.dialogue  

        for branch in self.sub_branches:
            path_dialogue = branch.find_dialogue_path(name)  
            if path_dialogue is not None:
                return self.dialogue + path_dialogue  
        return None  

    def add_branch(self, branch: "DialogueTree"):
        self.sub_branches.append(branch)

    def find_branch(self, name:str) -> "DialogueTree":
        if self.name == name:
            return self
        
        for branch in self.sub_branches:
            found_branch = branch.find_branch(name)
            
            if found_branch:
                return found_branch
        return None
    
    def is_leaf(self):
        return len(self.sub_branches) == 0
    
    def branch_prune(self, name:str):
        pass


class User(BaseModel):
    idx: int = None
    profile_description: str = ''
    #interaction_history: List[str] = [] # the interaction tweet content or maybe str directly?
    interaction_history: str = None # the aggregrated interaction history
    category_history: List[str] = [] # 
    dialogue_tree: DialogueTree = None # a tree structured dialogue history
    selected_leafs:Dict[str,List[int]] = {}# category:[leaf_ids] is this necessary?
    inter_nums:int = None
    #description: str = ""

if __name__ == '__main__':
    root = DialogueTree(name="root", type="start", dialogue=[LLMMessage(role = "user", content="Hello!")])
    node1 = DialogueTree(name="node1", type="response", dialogue=[LLMMessage(role = "user",content="How can I help you?")])
    node2 = DialogueTree(name="node2", type="response", dialogue=[LLMMessage(role = "user",content="What's your name?")])
    node3 = DialogueTree(name="node3", type="response", dialogue=[LLMMessage(role = "user",content="Nice to meet you!")])


    root.add_branch(node1)
    root.add_branch(node2)
    node1.add_branch(node3)
    # find the dialogue path
    merged_dialogue = root.find_dialogue_path("node3")
    if merged_dialogue:
        for message in merged_dialogue:
            print(message)
    else:
        print("Path not found")