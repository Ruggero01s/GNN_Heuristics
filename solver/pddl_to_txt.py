from unified_planning.io import PDDLReader

def translate_to_custom_encoding(pddl_problem):
    custom_encoding = []

    # BEGIN_TYPES
    custom_encoding.append("BEGIN_TYPES")
    type_id = 0
    type_mapping = {}  # Maps type names to their IDs
    # Iterate through all user-defined types
    
    
    type_mapping["object"] = type_id
    custom_encoding.append(f"{type_id} -1 object")
    
    type_id += 1

    for type in pddl_problem.user_types:
        if type.name != "object":
            type_mapping[type.name] = type_id
            custom_encoding.append(f"{type_id} {0 if type.father == None else type_mapping[type.father.name]} {type.name}")
            type_id += 1

    custom_encoding.append("END_TYPES")
    
    object_mapping = {}  # Maps object names to their IDs

    # BEGIN_OBJECTS
    custom_encoding.append("BEGIN_OBJECTS")
    for i, obj in enumerate(pddl_problem.all_objects):
        obj_type = obj.type.name
        type_id = type_mapping.get(obj_type)  
        object_mapping[obj.name] = i  # Map object name to its ID
        custom_encoding.append(f"{i} {type_id} {obj.name}")
    custom_encoding.append("END_OBJECTS")
    
    
    predicate_mapping = {}  # Maps predicate names to their IDs

   # BEGIN_PREDICATES
    custom_encoding.append("BEGIN_PREDICATES")
    for i, fluent in enumerate(pddl_problem.fluents):
        predicate_mapping[fluent.name] = i  # Map predicate name to its ID
        # Get the arity (number of arguments) of the predicate
        temp = f"{i} "
        for param in fluent.signature:
            param_type = param.type.name
            type_id = type_mapping.get(param_type)
            temp += (f"{type_id}") + (" ")
        temp += (fluent.name)
        custom_encoding.append(temp)
    custom_encoding.append("END_PREDICATES")
    
    
    
    # BEGIN_GOAL
    custom_encoding.append("BEGIN_GOAL")
    for goal in pddl_problem.goals:
        #TODO implement logic for OR goals??
        if goal.is_and():  # Handle AND goals 
            for subgoal in goal.args:
                fluent_name = subgoal.fluent().name
                if predicate_mapping[fluent_name] == 0: #TODO questo scempio di adhoc testing
                    temp = f"{predicate_mapping[fluent_name]} "
                    for arg in subgoal.args:
                        temp += f"{object_mapping[arg.object().name]}" + " "
                    custom_encoding.append(temp.rstrip())

        else:
            fluent_name = goal.fluent().name
            if predicate_mapping[fluent_name] == "on": #TODO questo scempio di adhoc testing
                temp = f"{predicate_mapping[fluent_name]} "
                for arg in goal.args:
                    temp += f"{object_mapping[arg.object().name]}" + " "
                    
            custom_encoding.append(temp.rstrip())
    custom_encoding.append("END_GOAL")


    # BEGIN_ACTIONS
    custom_encoding.append("BEGIN_ACTIONS")
    for action in pddl_problem.actions:
        custom_encoding.append(f"BEGIN_ACTION\n{action.name}")

        parameter_mapping = {}  # Maps predicate names to their IDs

        # BEGIN_PARAMETERS
        custom_encoding.append("BEGIN_PARAMETERS")
        for i, param in enumerate(action.parameters):
            param_type = param.type.name
            type_id = type_mapping.get(param_type)
            parameter_mapping[param.name] = i
            custom_encoding.append(f"{type_id} {type_mapping[param_type]}")  # The second value is always 0 (placeholder)
        custom_encoding.append("END_PARAMETERS")

        # BEGIN_PRECONDITION
        custom_encoding.append("BEGIN_PRECONDITION")
        for precond in action.preconditions:
            if precond.is_and():  # Handle AND preconditions
                for subprecond in precond.args:
                    fluent_name = subprecond.fluent().name
                    temp = f"{predicate_mapping[fluent_name]} "
                    for arg in subprecond.args:
                        temp += f"{parameter_mapping[arg.parameter().name]}" + " "
                    custom_encoding.append(temp.rstrip())

            else:
                fluent_name = precond.fluent().name
                temp = f"{predicate_mapping[fluent_name]} "
                for arg in precond.args:
                    temp += f"{parameter_mapping[arg.parameter().name]}" + " "
                custom_encoding.append(temp.rstrip())

        custom_encoding.append("END_PRECONDITION")
                
        # BEGIN_ADD_EFFECT
        custom_encoding.append("BEGIN_ADD_EFFECT")
        for effect in action.effects:
            if effect.value.is_true():
                fluent_name = effect.fluent.fluent().name
                temp = f"{predicate_mapping[fluent_name]} "
                for arg in effect.fluent.args:
                    temp += f"{parameter_mapping[arg.parameter().name]}" + " "
                custom_encoding.append(temp.rstrip())
        custom_encoding.append("END_ADD_EFFECTN")
        
        # BEGIN_DEL_EFFECT
        custom_encoding.append("BEGIN_DEL_EFFECT")
        for effect in action.effects:
            if effect.value.is_false():
                fluent_name = effect.fluent.fluent().name
                temp = f"{predicate_mapping[fluent_name]} "
                for arg in effect.fluent.args:
                    temp += f"{parameter_mapping[arg.parameter().name]}" + " "
                custom_encoding.append(temp.rstrip())
        custom_encoding.append("END_DEL_EFFECT")
        
        custom_encoding.append("END_ACTION")
    custom_encoding.append("END_ACTIONS")

    # BEGIN_STATIC_FACTS
    custom_encoding.append("BEGIN_STATIC_FACTS")
    #TODO Assuming no static facts for now
    custom_encoding.append("END_STATIC_FACTS")

    encoding = "\n".join(custom_encoding)
    return encoding, object_mapping, predicate_mapping


def append_state(state, object_mapping, predicate_mapping):
    temp_encoding = []
    temp_encoding.append("BEGIN_STATE_LIST")
    temp_encoding.append("BEGIN_LABELED_STATE")
    temp_encoding.append("-1")
    temp_encoding.append("BEGIN_STATE")
    for element in state:
        element=element.replace('(','').replace(')','')
        elements=element.split()
        temp = f"{predicate_mapping[elements[0]]} "
        for e in elements[1:]:
            temp += f"{object_mapping[e]}" + " "
        temp_encoding.append(temp.rstrip())
    temp_encoding.append("END_STATE")
    temp_encoding.append("END_LABELED_STATE")
    temp_encoding.append("END_STATE_LIST")

    
    return temp_encoding

# Example usage
#reader = PDDLReader()
#pddl_problem = reader.parse_problem('solver/data_pddl/transport/domain.pddl', 'solver/data_pddl/transport/p01.pddl')
#custom_encoding = translate_to_custom_encoding(pddl_problem)
#print(custom_encoding)