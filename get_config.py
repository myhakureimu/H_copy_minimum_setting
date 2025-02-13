def get_config(args):
    print(args.exp_name)
    if args.exp_name == 'IOHypothesis':
        if args.num_x == 5:
            table_lengths = [8]
            num_IO_h = [16, 16]  # 16 choose 8 = 12870
            if args.num_training_tables != 0:
                train_info = {8: args.num_training_tables}  # Number of train tables to sample per length
                testI_info = {8: 512}  # Number of test tables to sample per length
                testO_info = {8: 512}  # Number of test tables to sample per length
            else:
                train_info = {8: 12358}  # Number of train tables to sample per length
                testI_info = {8: 512}  # Number of test tables to sample per length
                testO_info = {8: 512}  # Number of test tables to sample per length
        elif args.num_x == 6:
            table_lengths = [8]
            num_IO_h = [args.num_training_hypotheses, 16]  # 48 choose 4 = 194580
            if args.num_training_tables != 0:
                train_info = {8: args.num_training_tables}  # Number of train tables to sample per length
                testI_info = {8: 0}  # Number of test tables to sample per length
                testO_info = {8: 512}  # Number of test tables to sample per length
            else:
                train_info = {8: 12358}  # Number of train tables to sample per length
                testI_info = {8: 0}  # Number of test tables to sample per length
                testO_info = {8: 512}  # Number of test tables to sample per length
        else:
            raise Exception('Setting Not Found')
    elif args.exp_name == 'IOHypothesis+Size':
        if args.num_x == 5:
            table_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            num_IO_h = [16, 16]  # 16 choose 8 = 12870
            if args.num_training_tables != 0:
                train_info = {
                    7: args.num_training_tables,
                    8: args.num_training_tables,
                    9: args.num_training_tables
                }  # Number of train tables to sample per length
            else:
                train_info = {7: 4096, 8: 4096, 9: 4096}  # Number of train tables to sample per length
            testI_info = {
                2: 120, 3: 512, 4: 512, 5: 512, 6: 512,
                7: 512, 8: 512, 9: 512,
                10:512, 11:512, 12:512, 13:512, 14:120,
                }  # Number of train tables to sample per length
            testO_info = {
                2: 120, 3: 512, 4: 512, 5: 512, 6: 512,
                7: 512, 8: 512, 9: 512,
                10:512, 11:512, 12:512, 13:512, 14:120,
                }  # Number of train tables to sample per length
        else:
            raise Exception('Setting Not Found')
    else:
        raise Exception('Setting Not Found')

    return table_lengths, num_IO_h, train_info, testI_info, testO_info
    # if split_based_on == 'table':
    #     if exp_name == 'IOHypothesis':
    #         if num_x == 5:
    #             table_lengths = [8]
    #             num_IO_h = [16, 16]  # 16 choose 8 = 12870
    #             train_info = {8: 4096}  # Number of train tables to sample per length
    #             testI_info = {8: 4096}  # Number of test tables to sample per length
    #             testO_info = {8: 4096}  # Number of test tables to sample per length
    #         else:
    #             raise Exception('Setting Not Found')
    #     elif exp_name == 'IDHypothesis':
    #         if num_x == 4:
    #             table_lengths = [4]
    #             split_ratio = [0.9, 0.1]  # Ratios for train and test splits
    #             train_info = {4: 1638}  # Number of train tables to sample per length
    #             test__info = {4: 182}  # Number of test tables to sample per length
    #         if num_x == 5:
    #             table_lengths = [8]
    #             split_ratio = [0.9, 0.1]  # Ratios for train and test splits
    #             train_info = {8: 1638}  # Number of train tables to sample per length
    #             test__info = {8: 182}  # Number of test tables to sample per length
    #         else:
    #             raise Exception('Setting Not Found')
    #         # if num_x == 5: 
    #         #     table_lengths = [4]
    #         #     split_ratio = [0.7, 0.3]  # Ratios for train and test splits
    #         #     train_info = {4: 1820}  # Number of train tables to sample per length
    #         #     test__info = {4: 1820}  # Number of test tables to sample per length
    #         # if num_x == 6:
    #         #     table_lengths = [4]
    #         #     split_ratio = [0.7, 0.3]  # Ratios for train and test splits
    #         #     train_info = {4: 1820}  # Number of train tables to sample per length
    #         #     test__info = {4: 1820}  # Number of test tables to sample per length
    #     elif exp_name == 'IDHypothesis+Size':
    #         # if num_x == 4:
    #         #     table_lengths = [4, 5, 6, 7]
    #         #     split_ratio = [0.7, 0.3]
    #         #     train_info = {4: 1638, 5: 1638, 6: 1638, 7:1638}  # Number of train tables to sample per length
    #         #     test__info = {4: 182, 5: 182, 6: 182, 7: 182}  # Number of train tables to sample per length
    #         if num_x == 5:
    #             table_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #             split_ratio = [1/3, 2/3]
    #             if args.num_training_tables != 0:
    #                 train_info = {
    #                     4: args.num_training_tables,
    #                     5: args.num_training_tables,
    #                     6: args.num_training_tables
    #                 }  # Number of train tables to sample per length
    #             else:
    #                 train_info = {4: 3000, 5: 3000, 6: 3000}  # Number of train tables to sample per length
    #             test__info = {
    #                 2: 512, 3: 512,
    #                 4: 512, 5: 512, 6: 512,
    #                 7: 512, 8: 512, 9: 512, 10: 512, 11: 512, 12: 300}  # Number of train tables to sample per length
    #         else:
    #             raise Exception('Setting Not Found')
    #     else:
    #         raise Exception('Setting Not Found')
    
    # if split_based_on == 'hypothesis':
    #     if exp_name == 'OODHypothesis':
    #         if num_x == 5:
    #             table_lengths = [4]
    #             split_ratio = [1/2, 1/2]  # Ratios for train and test splits
    #             train_info = {4: 1820}  # Number of train tables to sample per length
    #             test__info = {4: 1820}  # Number of test tables to sample per length
    #         else:
    #             raise Exception('Setting Not Found')
    #         # if num_x == 6:
    #         #     table_lengths = [4]
    #         #     split_ratio = [0.75, 0.25]  # Ratios for train and test splits
    #         #     train_info = None  # Number of train tables to sample per length
    #         #     test__info = None  # Number of test tables to sample per length
    #     elif exp_name == 'OODHypothesis+Size':
    #         if num_x == 6:
    #             table_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #             split_ratio = [1/2, 1/2]
    #             if args.num_training_tables != 0:
    #                 train_info = {
    #                     4: args.num_training_tables,
    #                     5: args.num_training_tables,
    #                     6: args.num_training_tables
    #                 }  # Number of train tables to sample per length
    #             else:
    #                 train_info = {4: 3000, 5: 3000, 6: 3000}  # Number of train tables to sample per length
    #             test__info = {
    #                 2: 512, 3: 512,
    #                 4: 512, 5: 512, 6: 512,
    #                 7: 512, 8: 512, 9: 512, 10: 512, 11: 512, 12: 300}  # Number of train tables to sample per length
    #         else:
    #             raise Exception('Setting Not Found')
    #     else:
    #         raise Exception('Setting Not Found')