from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from question_generators.dimensions.ethic.ethic_generator import generate_new_questions as ethic_generate
from question_generators.dimensions.ethic.ethic_validator import validate_and_select_best_question as ethic_validate

from question_generators.dimensions.fairness.fairness_generator import generate_new_questions as fairness_generate
from question_generators.dimensions.fairness.fairness_validator import validate_and_select_best_question as fairness_validate

from question_generators.dimensions.privacy.privacy_generator import generate_new_questions as privacy_generate
from question_generators.dimensions.privacy.privacy_validator import validate_and_select_best_question as privacy_validate

from question_generators.dimensions.safety.safety_generator import generate_new_questions as safety_generate
from question_generators.dimensions.safety.safety_validator import validate_and_select_best_question as safety_validate

from question_generators.dimensions.truth.truth_generator import generate_new_questions as truth_generate
from question_generators.dimensions.truth.truth_validator import validate_and_select_best_question as truth_validate


GeneratorFunc = Callable[..., List[Dict[str, Any]]]
ValidatorFunc = Callable[..., Optional[Dict[str, Any]]]


_DIMENSION_ROUTER: Dict[str, Dict[str, Callable[..., Any]]] = {
    "ethic": {
        "generate": ethic_generate,
        "validate": ethic_validate,
    },
    "fairness": {
        "generate": fairness_generate,
        "validate": fairness_validate,
    },
    "privacy": {
        "generate": privacy_generate,
        "validate": privacy_validate,
    },
    "safety": {
        "generate": safety_generate,
        "validate": safety_validate,
    },
}


def generate_and_validate_question(
    dimension: str,
    initial_question: str,
    current_question: str,
    model_answer: str,
    reference_answer: Optional[str] = None,
    history_questions: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """统一的新问题生成与验证入口。"""

    if history_questions is None:
        history_questions = []

    if dimension in _DIMENSION_ROUTER:
        generate_func: GeneratorFunc = _DIMENSION_ROUTER[dimension]["generate"]
        validate_func: ValidatorFunc = _DIMENSION_ROUTER[dimension]["validate"]

        new_questions = generate_func(
            initial_question=initial_question,
            current_question=current_question,
            model_answer=model_answer,
            history_questions=history_questions,
        )

        if not new_questions:
            return None

        return validate_func(
            current_question=current_question,
            new_questions=new_questions,
        )

    if dimension == "truth":
        def generate_func() -> List[Dict[str, Any]]:
            return truth_generate(
                initial_question=initial_question,
                current_question=current_question,
                model_answer=model_answer,
                reference_answer=reference_answer,
                history_questions=history_questions,
            )

        return truth_validate(
            current_question=current_question,
            generate_new_questions_func=generate_func,
        )

    raise ValueError(f"Unsupported dimension: {dimension}")
