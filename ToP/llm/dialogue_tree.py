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
    selected_leafs:Dict[str,List[int]] = {}
    inter_nums:int = None
    cat_value_counts: Dict[str, int] = None # category:count
    #description: str = ""
