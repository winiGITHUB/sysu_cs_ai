class Predicate:
    def __init__(self, str_in):
        self.elements = []  
        if len(str_in) != 0:
            if str_in[0] == ',':
                str_in = str_in[1:]
            tmp = ""
            for i in range(len(str_in)):
                tmp += str_in[i]
                if str_in[i] in ['(', ',', ')']:
                    self.elements.append(tmp[0:-1])  
                    tmp = ""

    def new(self, list_in):
        for element in list_in:
            self.elements.append(element)

    def rename(self, old_name, new_name):
        for i in range(len(self.elements)):
            for j in range(len(old_name)):
                if self.elements[i] == old_name[j]:
                    self.elements[i] = new_name[j]

    def get_pre(self):
        return self.elements[0][0] == "~"

    def get_name(self):
        if self.get_pre():
            return self.elements[0][1:]
        else:
            return self.elements[0]


def print_clause(clause_in):
    clause_str = ""
    if len(clause_in) > 1:
        clause_str += "("
    for i in range(len(clause_in)):
        clause_str += clause_in[i].elements[0]
        if len(clause_in[i].elements) > 1:
            clause_str += "("
            for j in range(1, len(clause_in[i].elements)):
                clause_str += clause_in[i].elements[j]
                if j < len(clause_in[i].elements) - 1:
                    clause_str += ","
            clause_str += ")"
        if i < len(clause_in) - 1:
            clause_str += ","
    if len(clause_in) > 1:
        clause_str += ")"
    return clause_str


def print_msg(key, i, j, old_name, new_name):
    msg = str(len(set_of_clause)) + ": R[" + str(i + 1)
    if len(new_name) == 0 and len(set_of_clause[i]) != 1:
        msg += chr(key + 97)
    msg += ", " + str(j + 1) + chr(key + 97) + "]("
    for k in range(len(old_name)):
        msg += old_name[k] + "=" + new_name[k]
        if k < len(old_name) - 1:
            msg += ", "
    msg += ") = "
    return msg


def end_or_not():
    for new_clause in set_of_clause:
        if not new_clause:
            return True
    return False


def main():
    global set_of_clause
    set_of_clause = []
    clauses = [
       "On(tony,mike)",
        "On(mike,john)",
        "Green(tony)",
        "~Green(john)",
        "(~On(x,y),~Green(x),Green(y))"
    ]

    for clause_in in clauses:
        if clause_in[0] == '(':
            clause_in = clause_in[1:-1]
        clause_in = clause_in.replace(' ', '')
        set_of_clause.append([])
        tmp = ""
        for j in range(len(clause_in)):
            tmp += clause_in[j]
            if clause_in[j] == ')':
                clause_tmp = Predicate(tmp)
                set_of_clause[-1].append(clause_tmp)
                tmp = ""

    for i in range(len(set_of_clause)):
        print(print_clause(set_of_clause[i]))

    status = True
    while status:
        for i in range(len(set_of_clause)):
            if not status:
                break
            if len(set_of_clause[i]) == 1:
                for j in range(len(set_of_clause)):
                    if not status:
                        break
                    if i == j:
                        continue
                    old_name = []
                    new_name = []
                    key = -1
                    for k in range(len(set_of_clause[j])):
                        if set_of_clause[i][0].get_name() == set_of_clause[j][k].get_name() and set_of_clause[i][0].get_pre() != set_of_clause[j][k].get_pre():
                            key = k
                            for l in range(1, len(set_of_clause[j][k].elements)):
                                if len(set_of_clause[j][k].elements[l]) == 1:
                                    old_name.append(set_of_clause[j][k].elements[l])
                                    new_name.append(set_of_clause[i][0].elements[l])
                                elif len(set_of_clause[i][0].elements[l]) == 1:
                                    old_name.append(set_of_clause[i][0].elements[l])
                                    new_name.append(set_of_clause[j][k].elements[l])
                                elif set_of_clause[j][k].elements[l] != set_of_clause[i][0].elements[l]:
                                    key = -1
                                    break
                            break
                    if key == -1:
                        continue
                    new_clause = []
                    for k in range(len(set_of_clause[j])):
                        if k != key:
                            p = Predicate("")
                            p.new(set_of_clause[j][k].elements)
                            p.rename(old_name, new_name)
                            new_clause.append(p)
                    if len(new_clause) == 1:
                        for k in range(len(set_of_clause)):
                            if len(set_of_clause[k]) == 1 and new_clause[0].elements == set_of_clause[k][0].elements:
                                key = -1
                                break
                    if key == -1:
                        continue
                    set_of_clause.append(new_clause)
                    print(print_msg(key, i, j, old_name, new_name), end="")
                    print(print_clause(new_clause))
                    if end_or_not():
                        status = False
                        break
            else:
                for j in range(len(set_of_clause)):
                    key = -1
                    if i != j and len(set_of_clause[i]) == len(set_of_clause[j]):
                        for k in range(len(set_of_clause[i])):
                            if set_of_clause[i][k].elements == set_of_clause[j][k].elements:
                                continue
                            elif set_of_clause[i][k].get_name() == set_of_clause[j][k].get_name() and set_of_clause[i][k].elements[1:] == set_of_clause[j][k].elements[1:]:
                                if key != -1:
                                    key = -1
                                    break
                                key = k
                            else:
                                key = -1
                                break
                    if key == -1:
                        continue
                    new_clause = []
                    for k in range(len(set_of_clause[i])):
                        if k != key:
                            p = Predicate("")
                            p.new(set_of_clause[j][k].elements)
                            new_clause.append(p)
                    if len(new_clause) == 1:
                        for k in range(len(set_of_clause)):
                            if len(set_of_clause[k]) == 1 and new_clause[0].elements == set_of_clause[k][0].elements:
                                key = -1
                                break
                    if key == -1:
                        continue
                    set_of_clause.append(new_clause)
                    print(print_msg(key, i, j, [], []), end="")
                    print(print_clause(new_clause))
                    if end_or_not():
                        status = False
                        break
        if not status:
            break

    print("Success!")


if __name__ == '__main__':
    main()
