from base import *


GENERATION_METHODS = {
    0: "Duyệt tất cả công thức",
    1: "Chỉ duyệt các công thức có các cụm con cùng cấu trúc"
}


INVESTMENT_METHODS = {
    0: '''Đầu tư công ty có value là value max của chu kì đầu tư, nếu có nhiều công ty có
          cùng value là value max thì không đầu tư.''',
    1: '''Đầu tư các công ty có value của năm đầu tư và năm trước đó đều vượt ngưỡng cho
          trước. Hoặc, đầu tư tất cả các công ty vượt ngưỡng nếu năm trước đó không có công
          ty nào vượt ngưỡng. Các trường hợp khác không đầu tư.''',
}


MEASUREMENT_METHODS = {
    0: "Geomean",
    1: "Harmean",
    2: "Geomean và độ chênh lệch giữa geomean và geo_limit",
    3: "Harmean và độ chênh lệch giữa harmean và har_limit",
}


class Generator(Base):
    def __init__(self,
                 data: pd.DataFrame,
                 generation_method: int = 0,
                 required_fields: list[str] = [],
                 multiple_cycles: bool = False,
                 investment_method: int = 0,
                 interest_rate: float = 1.00,
                 measurement_method: int = 0,
                 target: float = 1.0,
                 **kwargs) -> None:
        super().__init__(data)
        self.__last_cyc = self.data["TIME"].max()

        # Check generation_method
        if generation_method not in GENERATION_METHODS:
            raise Exception(f"Hiện chưa có cách sinh {generation_method}")

        self.generation_method = getattr(self, f"_Generator__generation_method_{generation_method}")

        # Check required fields
        self.required_fields = np.zeros(len(required_fields), np.int64)
        for i in range(len(required_fields)):
            for k, v in self.operand_name.items():
                if v == required_fields[i]:
                    self.required_fields[i] = k
                    break
            else:
                raise Exception(f"Không có trường {required_fields[i]} trong danh sách biến")

        self.required_fields.sort()
        self.required_fields = np.unique(self.required_fields)

        #
        self.multiple_cycles = multiple_cycles

        # Check investment_method
        if investment_method not in INVESTMENT_METHODS:
            raise Exception(f"Hiện chưa có cách đầu tư {investment_method}")

        self.investment_method = getattr(self, f"_Generator__investment_method_{investment_method}")

        #
        self.interest_rate = interest_rate

        # Check measurement_method
        if measurement_method not in MEASUREMENT_METHODS:
            raise Exception(f"Hiện chưa có cách đánh giá {measurement_method}")

        self.measurement_method = getattr(self, f"_Generator__measurement_method_{measurement_method}")

        #
        self.target = target

        # Các thuộc tính bổ sung
        for k, v in kwargs.items():
            setattr(self, k, v)

    def generate(self, path: str, num_f_per_file: int=1000, num_f_target: int=1000000):
        if not path.endswith("/") and not path.endswith("\\"):
            path += "/"

        if not os.path.exists(path):
            os.mkdir(path)
            print("Đã tạo thư mục lưu công thức")

        self.path = path
        print(path)

        self.list_f = []
        self.list_f_pro = []
        self.list_inv_cyc = []
        self.list_inv_pro = []
        self.generation_method(num_f_per_file, num_f_target)

    def __generation_method_1(self, num_f_per_file: int, num_f_target: int):
        try:
            self.history = list(np.load(self.path+"history.npy", allow_pickle=True))
        except:
            self.history = [np.zeros(2*max(len(self.required_fields), 1), np.int64), 0]

        self.current = [self.history[0].copy(), self.history[1]]
        self.count = np.array([0, num_f_per_file, 0, num_f_target])
        last_operand = self.current[0].shape[0] // 2
        num_operand = last_operand - 1

        while True:
            num_operand += 1
            print(f"Đang chạy sinh công thức, số toán hạng là {num_operand}")

            list_uoc_so = []
            for i in range(1, num_operand+1):
                if num_operand % i == 0:
                    list_uoc_so.append(i)

            start_divisor_idx = 0
            if num_operand == last_operand:
                start_divisor_idx = self.history[1]

            formula = np.full(num_operand*2, 0)
            for i in range(start_divisor_idx, len(list_uoc_so)):
                print("Số phần tử trong 1 cụm", list_uoc_so[i])
                struct = np.array([[0, list_uoc_so[i], 1+2*list_uoc_so[i]*j, 0] for j in range(num_operand//list_uoc_so[i])])
                if num_operand != last_operand or i != self.current[1]:
                    self.current[0] = formula.copy()
                    self.current[1] = i

                if self.required_fields.shape[0] == formula.shape[0] // 2:
                    sub_mode = True
                else:
                    sub_mode = False

                while self.__fill_1(formula, struct, 0, np.zeros(self.OPERAND.shape[1]), 0, np.zeros(self.OPERAND.shape[1]), 0, False, False, sub_mode, self.required_fields):
                    self.save_history()

            if self.save_history():
                break

        return

    def __fill_1(self, formula, struct, idx, temp_0, temp_op, temp_1, mode, add_sub_done, mul_div_done, sub_mode, check_op):
        if mode == 0: # Sinh dấu cộng trừ đầu mỗi cụm
            gr_idx = list(struct[:,2]-1).index(idx)

            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            for op in range(start, 2):
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                new_struct[gr_idx,0] = op
                if op == 1:
                    new_add_sub_done = True
                    new_formula[new_struct[gr_idx+1:,2]-1] = 1
                    new_struct[gr_idx+1:,0] = 1
                else:
                    new_add_sub_done = False

                if self.__fill_1(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, new_add_sub_done, mul_div_done, sub_mode, check_op):
                    return True
        elif mode == 2:
            start = 2
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            if start == 0:
                start = 2

            valid_op = func.get_valid_op(struct, idx, start)
            for op in valid_op:
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                if op == 3:
                    new_mul_div_done = True
                    for i in range(idx+2, 2*new_struct[0,1]-1, 2):
                        new_formula[i] = 3

                    for i in range(1, new_struct.shape[0]):
                        for j in range(new_struct[0,1]-1):
                            new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]
                else:
                    new_struct[:,3] += 1
                    new_mul_div_done = False
                    if idx == 2*new_struct[0,1] - 2:
                        new_mul_div_done = True
                        for i in range(1, new_struct.shape[0]):
                            for j in range(new_struct[0,1]-1):
                                new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]

                if self.__fill_1(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, add_sub_done, new_mul_div_done, sub_mode, check_op):
                    return True
        elif mode == 1:
            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            valid_operand = func.get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
            if sub_mode:
                valid_operand = np.intersect1d(valid_operand, check_op)

            if valid_operand.shape[0] > 0:
                if formula[idx-1] < 2:
                    temp_op_new = formula[idx-1]
                    temp_1_new = self.OPERAND[valid_operand].copy()
                else:
                    temp_op_new = temp_op
                    if formula[idx-1] == 2:
                        temp_1_new = temp_1 * self.OPERAND[valid_operand]
                    else:
                        temp_1_new = temp_1 / self.OPERAND[valid_operand]

                if idx + 1 == formula.shape[0] or (idx+2) in struct[:,2]:
                    if temp_op_new == 0:
                        temp_0_new = temp_0 + temp_1_new
                    else:
                        temp_0_new = temp_0 - temp_1_new
                else:
                    temp_0_new = np.array([temp_0]*valid_operand.shape[0])

                if idx + 1 != formula.shape[0]:
                    temp_list_formula = np.array([formula]*valid_operand.shape[0])
                    temp_list_formula[:,idx] = valid_operand
                    if idx + 2 in struct[:,2]:
                        if add_sub_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 0
                    else:
                        if mul_div_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 2

                    for i in range(valid_operand.shape[0]):
                        if valid_operand[i] in check_op:
                            new_check_op = check_op[check_op != valid_operand[i]]
                            new_sub_mode = sub_mode
                        else:
                            new_check_op = check_op.copy()
                            if idx + 1 + 2*check_op.shape[0] == formula.shape[0]:
                                new_sub_mode = True
                            else:
                                new_sub_mode = sub_mode

                        if self.__fill_1(temp_list_formula[i], struct, new_idx, temp_0_new[i], temp_op_new, temp_1_new[i], new_mode, add_sub_done, mul_div_done, new_sub_mode, new_check_op):
                            return True
                else:
                    temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                    temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308

                    formulas = np.array([formula]*valid_operand.shape[0])
                    formulas[:, idx] = valid_operand

                    self.count[0:3:2] += self.__handler(temp_0_new, formulas)
                    self.current[0][:] = formula[:]
                    self.current[0][idx] = self.OPERAND.shape[0]

                    if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                        return True

        return False

    def save_history(self):
        np.save(self.path+"history.npy", self.current)
        print("Đã lưu lịch sử")
        if self.count[0] == 0:
            return False

        df = pd.DataFrame({
            "formula": self.list_f,
            "f_profit": self.list_f_pro,
            "TIME": self.list_inv_cyc,
            "profit": self.list_inv_pro,
        })

        while True:
            path = self.path + f"formula_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv"
            if not os.path.exists(path):
                df.to_csv(path, index=False)
                self.count[0] = 0
                self.list_f = []
                self.list_f_pro = []
                self.list_inv_cyc = []
                self.list_inv_pro = []
                print("Đã lưu công thức")
                if self.count[2] >= self.count[3]:
                    raise Exception("Đã sinh đủ công thức")

                return False

    def __investment_method_0(self, weight, c_i):
        arr_index, arr_value, arr_profit = func.investment_method_0(weight, self.INDEX[c_i:] - self.INDEX[c_i], self.PROFIT[self.INDEX[c_i]:], self.interest_rate)
        return arr_index, arr_value, arr_profit

    def __measurement_method_0(self, weight, indexes, values, profits):
        if type(indexes[0]) == int or type(indexes[0]) == np.int64:
            f_profit = np.prod(profits[:-1])**(1.0/(len(profits) - 1))
            return f_profit, f_profit >= self.target
        elif type(indexes[0]) == list or type(indexes[0]) == np.ndarray:
            temp = np.zeros(len(profits)-1)
            for i in range(len(profits)-1):
                temp[i] = np.mean(profits[i])

            f_profit = np.prod(temp)**(1.0/len(temp))
            return f_profit, f_profit >= self.target

    def __handler(self, weights, formulas):
        count = 0
        if self.multiple_cycles:
            for w_i in range(weights.shape[0]):
                for c_i in range(self.number_cycle):
                    weight = weights[w_i][self.INDEX[c_i]:]
                    indexes, values, profits = self.investment_method(weight, c_i)
                    f_profit, check = self.measurement_method(weight, indexes, values, profits)
                    if check:
                        self.list_f.append(self.convert_arrF_to_strF(formulas[w_i]))
                        self.list_f_pro.append(f_profit)
                        self.list_inv_cyc.append(self.__last_cyc - c_i)
                        if type(indexes[0]) == int or type(indexes[0]) == np.int64:
                            self.list_inv_pro.append(profits[-1])
                        elif type(indexes[0]) == list or type(indexes[0]) == np.ndarray:
                            self.list_inv_pro.append(np.mean(profits[-1]))

                        count += 1
        else:
            for w_i in range(weights.shape[0]):
                indexes, values, profits = self.investment_method(weights[w_i], 0)
                f_profit, check = self.measurement_method(weights[w_i], indexes, values, profits)
                if check:
                    self.list_f.append(self.convert_arrF_to_strF(formulas[w_i]))
                    self.list_f_pro.append(f_profit)
                    self.list_inv_cyc.append(self.__last_cyc)
                    if type(indexes[0]) == int or type(indexes[0]) == np.int64:
                        self.list_inv_pro.append(profits[-1])
                    elif type(indexes[0]) == list or type(indexes[0]) == np.ndarray:
                        self.list_inv_pro.append(np.mean(profits[-1]))

                    count += 1

        return count

    def __generation_method_0(self, num_f_per_file: int, num_f_target: int):
        try:
            self.history = list(np.load(self.path+"history.npy", allow_pickle=True))
        except:
            num_operand = max(len(self.required_fields), 1)
            self.history = [
                np.array([
                    num_operand, # Số toán hạng trong công thức
                    0, # Số toán hạng trong các trừ cụm
                    0, # Cấu trúc các cộng cụm thứ mấy
                    0 # Cấu trúc các trừ cụm thứ mấy
                ]),
                None, # Cấu trúc công thức,
                None, # Công thức đã sinh đến
            ]

        self.current = [np.full(4, 0), None, None]
        self.count = np.array([0, num_f_per_file, 0, num_f_target])
        num_operand = self.history[0][0] - 1

        while True:
            num_operand += 1
            self.current[0][0] = num_operand
            print(f"Đang chạy sinh công thức, số toán hạng là {num_operand}")

            if self.current[0][0] == self.history[0][0]:
                start_num_sub_operand = self.history[0][1]
            else: start_num_sub_operand = 0

            for num_sub_operand in range(start_num_sub_operand, num_operand+1):
                self.current[0][1] = num_sub_operand
                temp_arr = np.full(num_sub_operand, 0)
                list_sub_struct = list([temp_arr])
                list_sub_struct.pop(0)
                func.split_posint_into_sum(num_sub_operand, temp_arr, list_sub_struct)

                num_add_operand = num_operand - num_sub_operand
                temp_arr = np.full(num_add_operand, 0)
                list_add_struct = list([temp_arr])
                list_add_struct.pop(0)
                func.split_posint_into_sum(num_add_operand, temp_arr, list_add_struct)

                if (self.current[0][0:2] == self.history[0][0:2]).all():
                    start_add_struct_idx = self.history[0][2]
                else: start_add_struct_idx = 0

                for add_struct_idx in range(start_add_struct_idx, len(list_add_struct)):
                    self.current[0][2] = add_struct_idx
                    if (self.current[0][0:3] == self.history[0][0:3]).all():
                        start_sub_struct_idx = self.history[0][3]
                    else: start_sub_struct_idx =  0

                    for sub_struct_idx in range(start_sub_struct_idx, len(list_sub_struct)):
                        self.current[0][3] = sub_struct_idx
                        add_struct = list_add_struct[add_struct_idx][list_add_struct[add_struct_idx]>0]
                        sub_struct = list_sub_struct[sub_struct_idx][list_sub_struct[sub_struct_idx]>0]
                        if type(self.history[1]) == type(None):
                            struct = func.create_struct(add_struct, sub_struct)
                            self.history[1] = struct
                        elif (self.current[0] == self.history[0]).all():
                            struct = self.history[1].copy()
                        else: struct = func.create_struct(add_struct, sub_struct)

                        self.current[1] = struct.copy()

                        while True:
                            if type(self.history[2]) == type(None):
                                formula = func.create_formula(struct)
                                self.history[2] = formula
                            elif struct.shape == self.history[1].shape and (struct == self.history[1]).all() and (self.current[0] == self.history[0]).all():
                                formula = self.history[2].copy()
                            else: formula = func.create_formula(struct)

                            self.current[2] = formula.copy()

                            if self.required_fields.shape[0] == formula.shape[0] // 2:
                                sub_mode = True
                            else:
                                sub_mode = False

                            while self.__fill_0(formula, struct, 1, np.zeros(self.OPERAND.shape[1]), -1, np.zeros(self.OPERAND.shape[1]), sub_mode, self.required_fields):
                                self.save_history()

                            if not func.update_struct(struct, self.numerator_condition):
                                break

            if self.save_history():
                break

        return

    def __fill_0(self, formula, struct, idx, temp_0, temp_op, temp_1, sub_mode, check_op):
        start = -1
        if (formula[0:idx] == self.current[2][0:idx]).all():
            start = self.current[2][idx]
        else:
            start = 0

        valid_operand = func.get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
        if sub_mode:
            valid_operand = np.intersect1d(valid_operand, check_op)

        if valid_operand.shape[0] > 0:
            if formula[idx-1] < 2:
                temp_op_new = formula[idx-1]
                temp_1_new = self.OPERAND[valid_operand].copy()
            else:
                temp_op_new = temp_op
                if formula[idx-1] == 2:
                    temp_1_new = temp_1 * self.OPERAND[valid_operand]
                else:
                    temp_1_new = temp_1 / self.OPERAND[valid_operand]

            if idx + 1 == formula.shape[0] or formula[idx+1] < 2:
                if temp_op_new == 0:
                    temp_0_new = temp_0 + temp_1_new
                else:
                    temp_0_new = temp_0 - temp_1_new
            else:
                temp_0_new = np.array([temp_0]*valid_operand.shape[0])

            if idx + 1 != formula.shape[0]:
                temp_list_formula = np.array([formula]*valid_operand.shape[0])
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                for i in range(valid_operand.shape[0]):
                    if valid_operand[i] in check_op:
                        new_check_op = check_op[check_op != valid_operand[i]]
                        new_sub_mode = sub_mode
                    else:
                        new_check_op = check_op
                        if idx + 1 + 2*check_op.shape[0] == formula.shape[0]:
                            new_sub_mode = True
                        else:
                            new_sub_mode = sub_mode

                    if self.__fill_0(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], new_sub_mode, new_check_op):
                        return True
            else:
                temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308

                formulas = np.array([formula]*valid_operand.shape[0])
                formulas[:, idx] = valid_operand

                self.count[0:3:2] += self.__handler(temp_0_new, formulas)
                self.current[2][:] = formula[:]
                self.current[2][idx] = self.OPERAND.shape[0]

                if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                    return True

        return False

    def __measurement_method_1(self, weight, indexes, values, profits):
        if type(indexes[0]) == int or type(indexes[0]) == np.int64:
            f_profit = func.harmean(profits[:-1])
            return f_profit, f_profit >= self.target
        elif type(indexes[0]) == list or type(indexes[0]) == np.ndarray:
            temp = np.zeros(len(profits)-1)
            for i in range(len(profits)-1):
                temp[i] = np.mean(profits[i])

            f_profit = func.harmean(temp)
            return f_profit, f_profit >= self.target

    def __investment_method_1(self, weight, c_i):
        INDEX = self.INDEX[c_i:] - self.INDEX[c_i]
        loop_threshold = weight[INDEX[-2]:INDEX[-1]]
        loop_threshold = np.unique(loop_threshold)
        loop_threshold[::-1].sort()
        if (loop_threshold <= -1.7976931348623157e+308).all():
            size = len(INDEX) - 2
            return [np.array([-1])]*size, [np.array([0.0])]*size, [np.array([0.0])]*size

        max_profit = -1.0
        list_index = []
        list_value = []
        list_profit = []
        count_loop = 0
        for threshold in loop_threshold:
            count_loop += 1
            temp_index = []
            temp_value = []
            temp_profit = []
            reason = 0
            for i in range(INDEX.shape[0]-2):
                inv_cyc_val = weight[INDEX[-i-3]:INDEX[-i-2]]
                inv_cyc_sym = self.SYMBOL[self.INDEX[-i-3]:self.INDEX[-i-2]]
                if reason == 0: # Không đầu tư do không có công ty nào vượt ngưỡng 2 năm liền
                    pre_cyc_val = weight[INDEX[-i-2]:INDEX[-i-1]]
                    pre_cyc_sym = self.SYMBOL[self.INDEX[-i-2]:self.INDEX[-i-1]]
                    a = np.where(pre_cyc_val > threshold)[0]
                    b = np.where(inv_cyc_val > threshold)[0]
                    coms = np.intersect1d(pre_cyc_sym[a], inv_cyc_sym[b])
                else:
                    b = np.where(inv_cyc_val > threshold)[0]
                    coms = inv_cyc_sym[b]

                if len(coms) == 0:
                    temp_index.append(np.array([-1]))
                    temp_value.append(np.array([0.0]))
                    temp_profit.append(np.array([self.interest_rate]))
                    if reason == 0 and b.shape[0] == 0:
                        reason = 1
                else:
                    index = np.where(np.isin(inv_cyc_sym, coms, True))[0]
                    value = weight[INDEX[-i-3]:INDEX[-i-2]][index]
                    profit = self.PROFIT[self.INDEX[-i-3]:self.INDEX[-i-2]][index]
                    index += INDEX[-i-3]
                    temp_index.append(index)
                    temp_value.append(value)
                    temp_profit.append(profit)
                    if reason == 1:
                        reason = 0

            total_profit, check = self.measurement_method(weight, temp_index, temp_value, temp_profit)
            if check and total_profit >= max_profit:
                max_profit = total_profit
                list_index = copy.deepcopy(temp_index)
                list_value = copy.deepcopy(temp_value)
                list_profit = copy.deepcopy(temp_profit)

            if count_loop == self.max_loop:
                break

        if max_profit == -1.0:
            size = len(INDEX) - 2
            return [np.array([-1])]*size, [np.array([0.0])]*size, [np.array([0.0])]*size

        return list_index, list_value, list_profit

    def __measurement_method_2(self, weight, indexes, values, profits):
        if type(indexes[0]) == int or type(indexes[0]) == np.int64:
            f_profit = np.prod(profits[:-1])**(1.0/(len(profits) - 1))
            if f_profit < self.target:
                return f_profit, False

            max_profit = func.measurement_method_2(indexes, values, profits, self.interest_rate, f_profit)
            return f_profit, max_profit - f_profit >= self.diff_p_p_lim

    def __measurement_method_3(self, weight, indexes, values, profits):
        if type(indexes[0]) == int or type(indexes[0]) == np.int64:
            f_profit = func.harmean(profits[:-1])
            if f_profit < self.target:
                return f_profit, False

            max_profit = func.measurement_method_3(indexes, values, profits, self.interest_rate, f_profit)
            return f_profit, max_profit - f_profit >= self.diff_p_p_lim
