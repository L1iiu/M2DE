import os
import json
from api_call import call_cuc_deepseek

class DialogueManager:
    """对话管理器 - 简化版本"""
    
    def __init__(self):
        self.max_recent_turns = 2
        self.max_summary_length = 120
        self.recent_dialogues = []
        self.old_summaries = []
        self.current_turn = 0
    
    def initialize_dialogue(self):
        """初始化对话"""
        self.recent_dialogues = []
        self.old_summaries = []
        self.current_turn = 0
    
    def add_turn(self, question, response, eval_result=None):
        """添加一轮对话并包含评估结果"""
        self.current_turn += 1
        
        current_turn_data = {
            "turn_number": self.current_turn,
            "question": question,
            "response": response,
            "voted_score": eval_result.get("voted_score", 0) if eval_result else 0,
            "eval_results": eval_result.get("eval_results", {}) if eval_result else {}
        }
        
        # 添加到最近对话
        self.recent_dialogues.append(current_turn_data)
        
        # 如果超过最大保留轮次，处理最旧的对话
        if len(self.recent_dialogues) > self.max_recent_turns:
            oldest_turn = self.recent_dialogues.pop(0)
            self._summarize_turn(oldest_turn)
        
        return self.get_combined_history()
    
    def _summarize_turn(self, turn_data):
        """摘要单轮对话"""
        prompt = self._build_summary_prompt(turn_data)
        
        try:
            system_prompt = "你是一个专业的对话摘要专家，能够准确概括对话的核心内容"
            summary = call_cuc_deepseek(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            clean_summary = summary.strip()
            if len(clean_summary) > self.max_summary_length:
                clean_summary = clean_summary[:self.max_summary_length] + "..."
                
            turn_summary = f"第{turn_data['turn_number']}轮摘要: {clean_summary}"
            self.old_summaries.append(turn_summary)
            
            if len(self.old_summaries) > 5:
                self.old_summaries = self.old_summaries[-5:]
                
        except Exception as e:
            q_preview = turn_data['question'][:30] + "..." if len(turn_data['question']) > 30 else turn_data['question']
            turn_summary = f"第{turn_data['turn_number']}轮: 用户询问{q_preview}"
            self.old_summaries.append(turn_summary)
    
    def _build_summary_prompt(self, turn_data):
        """构建摘要提示"""
        return f"""
请对以下对话轮次进行简要摘要：

【轮次信息】
第{turn_data['turn_number']}轮对话

【用户问题】
{turn_data['question']}

【模型回答】
{turn_data['response']}

【摘要要求】
1. 用一句话概括这轮对话的核心内容
2. 突出用户的主要意图
3. 保留关键信息点
4. 长度不超过{self.max_summary_length}字
5. 格式为: "第X轮摘要: [内容]"

请直接输出摘要内容：
"""
    
    def get_combined_history(self):
        """获取组合历史记录"""
        history_parts = []
        
        for summary in self.old_summaries:
            history_parts.append(summary)
        
        for dialogue in self.recent_dialogues:
            history_parts.append(f"用户(第{dialogue['turn_number']}轮): {dialogue['question']}")
            history_parts.append(f"你(第{dialogue['turn_number']}轮): {dialogue['response']}")
        
        if not history_parts:
            return "这是第一轮对话，没有历史记录。"
        
        return "\n".join(history_parts)
    
    def get_all_dialogues(self):
        """获取所有对话记录（包含评估结果）"""
        return self.recent_dialogues.copy()
    
    def get_current_turn(self):
        """获取当前轮次"""
        return self.current_turn