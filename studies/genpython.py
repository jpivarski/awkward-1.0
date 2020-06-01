import argparse
import pycparser
import re
import black

def preprocess(filename):
    code = ""
    func = False
    templ = False
    tokens = {}
    templateids = []
    templatecall = False
    with open(filename, "r") as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1].rstrip() + "\n"
            else:
                line = line.rstrip()
            if line.startswith("#"):
                continue
            if re.search("\/\/.*\n", line):
                line = re.sub("\/\/.*\n", "\n", line)
            if line.startswith("template") and func is False:
                templ = True
            if "delete []" in line:
                continue
            if "typename" in line:
                iterate = True
                tempids = []
                while iterate:
                    if re.search("typename [^,]*,", line) is not None:
                        tempids.append(line[re.search("typename [^,]*,", line).span()[0]+9:re.search("typename [^,]*,", line).span()[1]-1])
                        line = line[re.search("typename [^,]*,", line).span()[1]:]
                    if re.search("typename [^,]*,", line) is None:
                        iterate = False
                if re.search("typename [^,]*>", line) is not None:
                    tempids.append(line[re.search("typename [^,]*>", line).span()[0]+9:re.search("typename [^,]*>", line).span()[1]-1])
                    line = line[re.search("typename [^,]*>", line).span()[1]:]
                for x in tempids:
                    templateids.append(x + "*")
                for x in tempids:
                    templateids.append(x)
                continue
            if func is True and line.count("{") > 0:
                for _ in range(line.count("{")):
                    parans.append("{")
            if func is False and re.search("\s.*\(", line):
                funcname = re.search("\s.*\(", line).group()[1:-1]
                tokens[funcname] = {"type": line.lstrip().split(" ")[0]}
                line = line.replace(line.split(" ")[0], "int")
                func = True
                parans = []
                code += line
                if line.count("{") > 0:
                    for _ in range(line.count("{")):
                        parans.append("{")
                continue
            if func is True and re.search("<.*>", line) is not None:
                line = line.replace(re.search("<.*>", line).group(), "")
            elif func is True and re.search("<.*\n", line) is not None and ";" not in line and ")" not in line and "[" not in line:
                templatecall = True
                line = line.replace(re.search("<.*\n", line).group(), "")
            elif func is True and re.search(".*>", line) is not None and ";" not in line and "(" not in line[:re.search(".*>", line).span()[1]]:
                line = line.replace(re.search(".*>", line).group(), "")
                templatecall = False
            elif func is True and templatecall is True:
                line = ""
            if func is True and re.search("[\W_]*=[\W_]*new u?int\d{1,2}_t\[.\];", line) is not None:
                line = line.replace(re.search("[\W_]*=[\W_]*new u?int\d{1,2}_t\[.\];", line).group(), ";")
            if func is True and re.search("u?int\d{1,2}_t\*?", line) is not None:
                line = line.replace(re.search("u?int\d{1,2}_t", line).group(), "int")
            if func is True and templ is True:
                for x in templateids:
                    if x in line:
                        if line[line.find(x)-1] == " " or line[line.find(x)-1] == "*" or line[line.find(x)-1]:
                            if x.endswith("*"):
                                x = x[:-1]
                            line = line.replace(x, "int")
            if func is True and line.find("bool") != -1:
                if line.find("bool*") != -1:
                    typename = "bool*"
                else:
                    typename = "bool"
                if "=" not in line and "(" not in line:
                    varname = line[line.find(typename) + len(typename) + 1:]
                    varname = re.sub("[\W_]+", "", varname)
                    tokens[funcname][varname] = "bool"
                line = line.replace("bool", "int", 1)
            code += line
            if func is True and line.count("}") > 0:
                for _ in range(line.count("}")):
                    parans.pop()
                if len(parans) == 0:
                    func = False
                    templ = False
                    templateids = []

    return code,tokens

class FuncBody(object):

    def __init__(self, ast):
        self.ast = ast
        self.code = ""
        self.traverse(self.ast.block_items, 4)

    def traverse(self, item, indent, called=False):
        if item.__class__.__name__ == "list":
            for node in item:
                self.traverse(node, indent)
        elif item.__class__.__name__ == "Return":
            if item.expr.name.name == "failure":
                stmt = " "*indent + "raise ValueError({0})".format(item.expr.args.exprs[0].value)
            else:
                stmt = " "*indent + "return {0}".format(self.traverse(item.expr, 0, called=True))
            if called:
                return stmt
            else:
                self.code += stmt + "\n"
        elif item.__class__.__name__ == "Constant":
            constant = " "*indent + "{0}".format(item.value)
            if called:
                return constant
            else:
                self.code += constant
        elif item.__class__.__name__ == "Decl":
            if item.init is not None:
                stmt = " "*indent + "{0} = {1}".format(item.name, self.traverse(item.init, 0, called=True))
                if not called:
                    stmt = stmt + "\n"
            elif item.type.__class__.__name__ == "PtrDecl":
                stmt = " "*indent + "{0} = []".format(item.name)
                if not called:
                    stmt = stmt + "\n"
            else:
                stmt = ""
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Assignment":
            assignstmt = " "*indent + "{0} = {1}".format(self.traverse(item.lvalue, 0, called=True), self.traverse(item.rvalue, 0, called=True))
            if called:
                return assignstmt
            else:
                self.code += assignstmt + "\n"
        elif item.__class__.__name__ == "FuncCall":
            if item.args is not None:
                return " "*indent + "{0}({1})".format(item.name.name, self.traverse(item.args, 0, called=True))
            else:
                return " " * indent + "{0}()".format(item.name.name)
        elif item.__class__.__name__ == "ExprList":
            exprlist = " "*indent
            for i in range(len(item.exprs)):
                if i == 0:
                    exprlist += "{0}".format(self.traverse(item.exprs[i], 0, called=True))
                else:
                    exprlist += ", {0}".format(self.traverse(item.exprs[i], 0, called=True))
            if called:
                return exprlist
            else:
                self.code += exprlist
        elif item.__class__.__name__ == "BinaryOp":
            if item.op == "&&":
                operator = "and"
            elif item.op == "||":
                operator = "or"
            else:
                operator = item.op
            binaryopl = "{0}".format(self.traverse(item.left, 0, called=True))
            binaryopr = "{0}".format(self.traverse(item.right, 0, called=True))
            if called and item.left.__class__.__name__ == "BinaryOp":
                binaryopl = "("+binaryopl+")"
            if called and item.right.__class__.__name__ == "BinaryOp":
                binaryopr = "("+binaryopr+")"
            binaryop = " "*indent + "{0} {1} {2}".format(binaryopl, operator, binaryopr)
            if called:
                return binaryop
            else:
                self.code += binaryop
        elif item.__class__.__name__ == "If":
            ifstmt = " "*indent + "if {0}:\n".format(self.traverse(item.cond, 0, called=True))
            ifstmt += "{0}\n".format(self.traverse(item.iftrue, indent + 4, called=True))
            if item.iffalse is not None:
                ifstmt += " "*indent + "else:\n"
                ifstmt += "{0}\n".format(self.traverse(item.iffalse, indent + 4, called=True))
            if called:
                return ifstmt[:-1]
            else:
                self.code += ifstmt
        elif item.__class__.__name__ == "For":
            if (item.init is not None) and (item.next is not None) and (item.cond is not None) and (item.init.decls[0].init.__class__.__name__ == "Constant") and (len(item.init.decls) == 1) and (item.init.decls[0].init.value == "0") and (item.next.op == "p++") and (item.cond.op == "<") and (item.cond.left.name == item.init.decls[0].name):
                forstmt = " "*indent + "for {0} in range({1}):\n".format(item.init.decls[0].name, self.traverse(item.cond.right, 0, called=True))
                for i in range(len(item.stmt.block_items)):
                    forstmt += self.traverse(item.stmt.block_items[i], indent+4, called=True) + "\n"
            else:
                if item.init is not None:
                    forstmt = "{0}\n".format(self.traverse(item.init, indent, called=True))
                else:
                    forstmt = ""
                forstmt += " "*indent + "while {0}:\n".format(self.traverse(item.cond, 0, called=True))
                for i in range(len(item.stmt.block_items)):
                    forstmt += self.traverse(item.stmt.block_items[i], indent+4, called=True) + "\n"
                forstmt += " "*(indent+4) + "{0}\n".format(self.traverse(item.next, 0, called=True))
            if called:
                return forstmt[:-1]
            else:
                self.code += forstmt
        elif item.__class__.__name__ == "UnaryOp":
            if item.op[1:] == "++":
                unaryop = " "*indent + "{0} = {0} + 1".format(self.traverse(item.expr, 0, called=True))
            elif item.op[1:] == "--":
                unaryop = " " * indent + "{0} = {0} - 1".format(self.traverse(item.expr, 0, called=True))
            elif item.op == "*":
                unaryop = " "*indent + "{0}".format(self.traverse(item.expr, 0, called=True))
            elif item.op == "-":
                unaryop = " "*indent + "-{0}".format(self.traverse(item.expr, 0, called=True))
            elif item.op == "!":
                unaryop = " "*indent + "not {0}".format(self.traverse(item.expr, 0, called=True))
            else:
                raise NotImplementedError("Unhandled Unary Operator case. Please inform the developers about the error")
            if called:
                return unaryop
            else:
                self.code += unaryop + "\n"
        elif item.__class__.__name__ == "DeclList":
            decllist = " "*indent
            for i in range(len(item.decls)):
                if i == 0:
                    decllist += "{0}".format(self.traverse(item.decls[i], 0, called=True))
                else:
                    decllist += ", {0}".format(self.traverse(item.decls[i], 0, called=True))
            if called:
                return decllist
            else:
                self.code += decllist + "\n"
        elif item.__class__.__name__ == "ArrayRef":
            if item.subscript.__class__.__name__ == "UnaryOp":
                arrayref = " "*indent + "{0};".format(self.traverse(item.subscript, 0, called=True)) + " {0}[{1}]".format(self.traverse(item.name, 0, called=True), self.traverse(item.subscript.expr, 0, called=True))
            else:
                arrayref = " "*indent + "{0}[{1}]".format(self.traverse(item.name, 0, called=True), self.traverse(item.subscript, 0, called=True))
            if called:
                return arrayref
            else:
                self.code += arrayref
        elif item.__class__.__name__ == "Cast":
            cast = " "*indent + "{0}({1})".format(self.traverse(item.to_type, 0, called=True), self.traverse(item.expr, 0, called=True))
            if called:
                return cast
            else:
                self.code += cast
        elif item.__class__.__name__ == "Typename":
            typename = " "*indent + "{0}".format(item.type.type.names[0])
            if called:
                return typename
            else:
                self.code += typename
        elif item.__class__.__name__ == "ID":
            ID = " "*indent + "{0}".format(item.name)
            if called:
                return ID
            else:
                self.code += ID
        elif item.__class__.__name__ == "Compound":
            compound = ""
            if called:
                for i in range(len(item.block_items)):
                    compound += self.traverse(item.block_items[i], indent, called=True) + "\n"
            else:
                for i in range(len(item.block_items)):
                    compound += self.traverse(item.block_items[i], indent + 4, called=True) + "\n"
            return compound[:-1]
        elif item.__class__.__name__ == "TernaryOp":
            stmt = " "*indent + "{0} if {1} else {2}".format(self.traverse(item.iftrue, 0, called=True), self.traverse(item.cond, 0, called=True), self.traverse(item.iffalse, 0, called=True))
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "While":
            forstmt = " " * indent + "while {0}:\n".format(self.traverse(item.cond, 0, called=True))
            for i in range(len(item.stmt.block_items)):
                forstmt += self.traverse(item.stmt.block_items[i], indent + 4, called=True) + "\n"
            if called:
                return forstmt
            else:
                self.code += forstmt
        elif item.__class__.__name__ == "EmptyStatement":
            pass
        else:
            raise Exception("Unable to parse {0}".format(item.__class__.__name__))

class FuncDecl(object):

    def __init__(self, ast):
        self.ast = ast
        self.name = ast.name
        self.args = []
        self.returntype = self.ast.type.type.type.names[0]
        self.traverse()

    def traverse(self):
        if self.ast.type.args is not None:
            params = self.ast.type.args.params
            for param in params:
                typename, listcount = self.iterclass(param.type, 0)
                self.args.append({"name": param.name,
                                  "type": typename,
                                  "list": listcount})

    def iterclass(self, obj, count):
        if obj.__class__.__name__ == "IdentifierType":
            return obj.names[0], count
        elif obj.__class__.__name__ == "TypeDecl":
            return self.iterclass(obj.type, count)
        elif obj.__class__.__name__ == "PtrDecl":
            return self.iterclass(obj.type, count+1)

    def arrange_args(self):
        arranged = ""
        for i in range(len(self.args)):
            if i != 0:
                arranged += ", "
            arranged += "{0}: ".format(self.args[i]["name"]) + "List["*self.args[i]["list"] + self.args[i]["type"] + "]"*self.args[i]["list"]
        return arranged

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("filename")
args = arg_parser.parse_args()
filename = args.filename

if __name__ == "__main__":
    pfile, tokens = preprocess(filename)
    ast = pycparser.c_parser.CParser().parse(pfile)
    # Initialize black config
    blackmode = black.FileMode()
    for i in range(len(ast.ext)):
        decl = FuncDecl(ast.ext[i].decl)
        body = FuncBody(ast.ext[i].body)
        print(decl.name)
        print("----------------------------------------------------")
        print()
        funcgen = "def {0}({1}):\n".format(decl.name, decl.arrange_args())
        funcgen += body.code
        gencode = black.format_str(funcgen, mode=blackmode)
        print(gencode)
        print()
