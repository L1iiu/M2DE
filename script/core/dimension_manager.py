import importlib
import os

class DimensionManager:
    """维度模块管理器 - 负责加载各维度的专用模块，无回退机制"""
    
    @staticmethod
    def get_generator(dimension):
        """获取指定维度的生成器 - 必须存在专用模块"""
        try:
            module_path = f"dimension_modules.{dimension}.question_generator"
            module = importlib.import_module(module_path)
            generator = module.QuestionGenerator()
            print(f"✅ 成功加载 {dimension} 维度专用生成器")
            return generator
        except ImportError as e:
            raise ImportError(f"❌ 无法加载 {dimension} 维度的生成器: {e}。请确保该维度的专用模块已创建。")
    
    @staticmethod
    def get_validator(dimension):
        """获取指定维度的验证器 - 必须存在专用模块"""
        try:
            module_path = f"dimension_modules.{dimension}.question_validator"
            module = importlib.import_module(module_path)
            validator = module.QuestionValidator()
            print(f"✅ 成功加载 {dimension} 维度专用验证器")
            return validator
        except ImportError as e:
            raise ImportError(f"❌ 无法加载 {dimension} 维度的验证器: {e}。请确保该维度的专用模块已创建。")
    
    @staticmethod
    def get_script_path(dimension, script_id):
        """获取剧本文件路径 - 只从维度专用目录查找"""
        script_dir = f"dimension_modules/{dimension}/scripts"
        
        if not os.path.exists(script_dir):
            raise FileNotFoundError(f"❌ 维度 {dimension} 的剧本目录不存在: {script_dir}")
        
        # 先尝试精确匹配
        json_path = os.path.join(script_dir, f"{script_id}.json")
        if os.path.exists(json_path):
            return json_path
        
        # 尝试前缀匹配
        for file in os.listdir(script_dir):
            if file.startswith(script_id) and file.endswith('.json'):
                found_path = os.path.join(script_dir, file)
                print(f"📁 找到前缀匹配的剧本: {found_path}")
                return found_path
        
        raise FileNotFoundError(f"❌ 在维度 {dimension} 中未找到剧本: {script_id}")