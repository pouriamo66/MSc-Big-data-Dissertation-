##
## COLUMNS ARE INPUT (HISTORY AND NEIGHBOUR EFFECT) AND OUTPUT CHOICE
##

## INPUT CELLS:   Cell_1_history, Cell_2_history, ... Cell_9_history
##                Cell_1_neigh_e, Cell_1_neigh_e, ... Cell_1_neigh_e

## OUTPUT CELLS:  Cell_1_choice, Cell_2_choice, Cell_3_choice

##
## ROWS ARE INDIVIDUAL DECISIONS, ASSUMED TO BE INDEPENDENT
##

#################################################
#               #               #               #
#               #               #               #
#       1       #       2       #      3        #
#               #               #               #
#               #               #               #
#################################################
#               #               #               #
#               #               #               #
#      4        #      5        #      6        #
#               #               #               #
#               #               #               #
#################################################
#               #               #               #
#               #               #               #
#      7        #      8        #      9        #
#               #               #               #
#               #               #               #
################################################################################
#
#
#
#
#
#

prc <- process_mada_csv_files("mada_data");
mrb <- mada_row_based(prc, filename = "mada_rows.csv");

dat <- read.csv("mada_rows.csv");
dat <- retain_T1_T2(dat);
dat <- retain_ch_hist(dat);
dat <- create_hist_col(dat);

dat <- clean_choice(dat);
dat <- clean_hist(dat);
dat <- clean_neig(dat);

dat$c1 <- merge_choice(dat$c1);
dat$c2 <- merge_choice(dat$c2);
dat$c3 <- merge_choice(dat$c3);

piv_dat <- pivot_data(dat);
piv_dat <- pivot_colnames(piv_dat);

write.csv(piv_dat, file = "farm_or_fallow.csv");


pivot_colnames <- function(piv_dat){
    colnames(piv_dat) <- c("I_Subsidy",
                           "I_Neigh_c1",
                           "I_Neigh_c2",
                           "I_Neigh_c3",
                           "I_Neigh_c4",
                           "I_Neigh_c5",
                           "I_Neigh_c6",
                           "I_Neigh_c7",
                           "I_Neigh_c8",
                           "I_Neigh_c9",
                           "I_Hist_c1",
                           "I_Hist_c2",
                           "I_Hist_c3",
                           "I_Hist_c4",
                           "I_Hist_c5",
                           "I_Hist_c6",
                           "I_Hist_c7",
                           "I_Hist_c8",
                           "I_Hist_c9", 
                           "O_Choice_c1",
                           "O_Choice_c2",
                           "O_Choice_c3",
                           "O_Choice_c4",
                           "O_Choice_c5",
                           "O_Choice_c6",
                           "O_Choice_c7",
                           "O_Choice_c8",
                           "O_Choice_c9");
    return(piv_dat);
}


retain_T1_T2 <- function(dat){
    dat_T1 <- dat[dat[,2] == 1,];
    dat_T2 <- dat[dat[,2] == 2,];
    dat    <- rbind(dat_T1, dat_T2);
    return(dat);
}

create_hist_col <- function(dat){
    c1 <- which(grepl("Cell_choice", dat[,1]) == TRUE);
    c2 <- which(grepl("Cell_hist", dat[,1]) == TRUE);
    c3 <- which(grepl("Cell_neig", dat[,1]) == TRUE);
    if(length(c1) != length(c2) | length(c1) != length(c3)){
        stop("Error in cell choice and history lengths");
    }
    return(list(c1 = dat[c1,], c2 = dat[c2,], c3 = dat[c3,]));
}

retain_ch_hist <- function(dat){
    rows_ch <- grepl("Cell_choice", dat[,1]);
    rows_hi <- grepl("Cell_hist", dat[,1]);
    rows_ng <- grepl("Cell_neig", dat[,1]);
    rows    <- rows_ch | rows_hi | rows_ng;
    dat     <- dat[rows,];
    dat     <- dat[!is.na(dat[,4]),];
    return(dat);
}


clean_choice <- function(dat){
    cvec <- rep(x = NA, length = dim(dat$c1)[1]);
    for(i in 1:36){
        cell       <- paste("Cell_choice_", i , sep = "");
        repl       <- which(dat$c1[,1] == cell);
        cvec[repl] <- i;
    }
    dat$c1[,1]          <- cvec;
    colnames(dat$c1)[1] <- "Cell"
    return(dat);
}

clean_hist <- function(dat){
    cvec <- rep(x = NA, length = dim(dat$c2)[1]);
    for(i in 1:36){
        cell       <- paste("Cell_hist_", i , sep = "");
        repl       <- which(dat$c2[,1] == cell);
        cvec[repl] <- i;
    }
    dat$c2[,1]          <- cvec;
    colnames(dat$c2)[1] <- "Cell"
    return(dat);
}

clean_neig <- function(dat){
    cvec <- rep(x = NA, length = dim(dat$c3)[1]);
    for(i in 1:36){
        cell       <- paste("Cell_neig_", i , sep = "");
        repl       <- which(dat$c3[,1] == cell);
        cvec[repl] <- i;
    }
    dat$c3[,1]          <- cvec;
    colnames(dat$c3)[1] <- "Cell"
    return(dat);
}


correct_cells <- function(dat, col = 1){
    dat[which(dat[,col] %in% c(1,  6,  31, 36)), col] <- -1;
    dat[which(dat[,col] %in% c(2,  5,  32, 35)), col] <- -2;
    dat[which(dat[,col] %in% c(3,  4,  33, 34)), col] <- -3;
    dat[which(dat[,col] %in% c(7,  12, 25, 30)), col] <- -4;
    dat[which(dat[,col] %in% c(8,  11, 26, 29)), col] <- -5;
    dat[which(dat[,col] %in% c(9,  10, 27, 28)), col] <- -6;
    dat[which(dat[,col] %in% c(13, 18, 19, 24)), col] <- -7;
    dat[which(dat[,col] %in% c(14, 17, 20, 23)), col] <- -8;
    dat[which(dat[,col] %in% c(15, 16, 21, 22)), col] <- -9;
    dat[,col] <- abs(dat[,col]);
    return(dat);
}

merge_choice <- function(dat){
    dat[,4] <- apply(X = dat[,4:7], MARGIN = 1, FUN = sum);
    dat     <- dat[,1:4];
    return(dat);
}

pivot_data <- function(dat){
    rnds  <- dim(dat$c1)[1] / 36;
    pivot <- matrix(data = 0, nrow = rnds * 4, ncol = 28);
    start <- 1;
    end   <- 36;
    prow  <- 1;
    while(end <= dim(dat$c1)[1]){
        c1_rnd <- dat$c1[start:end,];
        c2_rnd <- dat$c2[start:end,];
        c3_rnd <- dat$c3[start:end,];
        al_rnd <- cbind(c1_rnd, c2_rnd, c3_rnd);
        p1     <- correct_cells(al_rnd[c(1:3, 7:9, 13:15),]);
        p2     <- correct_cells(al_rnd[c(4:6, 10:12, 16:18),]);
        p3     <- correct_cells(al_rnd[c(19:21, 25:27, 31:33),]);
        p4     <- correct_cells(al_rnd[c(22:24, 28:30, 34:36),]);
        p1     <- p1[order(p1[,1]),];
        p2     <- p2[order(p2[,1]),]
        p3     <- p3[order(p3[,1]),]
        p4     <- p4[order(p4[,1]),]
        pivot  <- pivot_row(pivot, prow, pp = p1);
        prow   <- prow + 1;
        pivot  <- pivot_row(pivot, prow, pp = p2);
        prow   <- prow + 1;
        pivot  <- pivot_row(pivot, prow, pp = p3);
        prow   <- prow + 1;
        pivot  <- pivot_row(pivot, prow, pp = p4);
        prow   <- prow + 1;
        start  <- end   + 1;
        end    <- start + 35;
    }
    return(pivot);
}

pivot_row <- function(pivot, prow, pp){
    pivot[prow, 1]      <- pp[1, 2];
    pivot[prow, 20:28]  <- pp[,4];
    pivot[prow, 11:19]  <- pp[,8];
    pivot[prow, 2:10]   <- pp[,12];
    return(pivot);
}

######################################################

mada_row_based <- function(process_csv_files_output, game_out_rows = 193,
                           max_rounds = 9, filename = "mada_rows.csv"){
    # First make the table
    games      <- process_csv_files_output$games;
    game_n     <- unique(games);
    game_rows  <- rep(x = 0, times = length(game_n));
    game_rows[game_n == "P"] <- 3 * game_out_rows;
    game_rows[game_n != "P"] <- max_rounds * game_out_rows;
    table_rows <- sum(game_rows); # Rows for the big table
    results    <- process_csv_files_output$results;
    player_lst <- NULL;
    for(i in 1:length(results)){
        game_vec        <- rep(games[i], length = dim(results[[i]])[1]);
        game_num        <- as.numeric(get_game_type(game_vec));
        results[[i]]    <- cbind(game_num, results[[i]]);
    }
    tabl <- do.call("rbind", results);
    tabl <- tabl[tabl[,2] <= max_rounds,];
    colnames(tabl) <- c("Game", "Round", "Player_1", "Player_2", "Player_3",
                        "Player_4");
    write.csv(tabl, filename);
    return(tabl)
}

process_mada_csv_files <- function(dir = getwd()){
    all_csvs   <- list.files(path = dir, pattern = ".csv");
    num_csvs   <- length(all_csvs);
    games      <- NULL;
    results    <- NULL;
    times      <- NULL;
    culled     <- NULL;
    list_el  <- 1;
    for(i in 1:num_csvs){
        filename <- all_csvs[i];
        if( identical(dir, getwd()) == FALSE ){
            filename <- paste(dir,"/", all_csvs[i], sep= "");
        }
        check_file <- scan(file = filename, what = "character");
        if(length(check_file) < 2){
            print(paste("Error in ", filename));
            break;
        }
        if(check_file[1] == "Player" & check_file[2] == "1"){
            file_res             <- summarise_mada(filename);
            game_type            <- file_res[[1]];
            game_res             <- file_res[[2]];
            games                <- c(games, game_type);
            results[[list_el]]   <- game_res;
            list_el              <- list_el + 1;
        }
    }
    return(list(games = games, results = results));
}


get_game_type <- function(game_vec){
    game_vec[game_vec == "P"]  <- 0;
    game_vec[game_vec == "T1"] <- 1;
    game_vec[game_vec == "T2"] <- 2;
    game_vec[game_vec == "T3"] <- 3;
    game_vec[game_vec == "T4"] <- 4;
    game_vec[game_vec == "T5"] <- 5;
    return(game_vec);
}

summarise_mada <- function(filename){
    dat         <- scan(file = filename, what = "character");
    tags        <- which(dat == "Tag:");
    game_types  <- dat[tags + 1];
    rounds      <- get_round_number(dat, tags, 1);
    Results     <- NULL;
    subsidy_lev <- get_subsidy_level(dat);
    time_table  <- get_round_times(dat);
    cell_hist   <- matrix(data = 0, nrow = 36, ncol = 4);
    for(i in 1:rounds){
        choices    <- get_choices(dat, round = i);
        taps       <- get_taps(dat, round = i);
        cell_sum   <- cell_summary(choices, taps);
        
        scores     <- get_scores(dat, round = i);
        subsidy    <- rep(x = subsidy_lev, length = dim(scores)[2]);
        year       <- rep(x = time_table[i, 4], length = dim(scores)[2]);
        month      <- rep(x = time_table[i, 3], length = dim(scores)[2]);
        day        <- rep(x = time_table[i, 2], length = dim(scores)[2]);
        hour       <- rep(x = time_table[i, 5], length = dim(scores)[2]);
        minute     <- rep(x = time_table[i, 6], length = dim(scores)[2]);
        second     <- rep(x = time_table[i, 7], length = dim(scores)[2]);
        ch_mat     <- cbind(choices, choices, choices, choices);
        hist_mat   <- matrix(data = 0, nrow = 36, ncol = 4);
        neig_mat   <- matrix(data = 0, nrow = 36, ncol = 4);
        rownames(hist_mat) <- paste("Cell_hist_", 1:36, sep = "");
        rownames(neig_mat) <- paste("Cell_neig_", 1:36, sep = "");
        
        
        game_order <- rep(x = 0, length = dim(scores)[2]);
        rres       <- rbind(scores, subsidy, cell_sum, ch_mat, hist_mat, 
                            neig_mat, year, month, day, hour, minute, second, 
                            game_order);
        rname      <- rep(x = i, times = dim(rres)[1]);
        Rndinfo    <- cbind(rname, rres); 
        Results    <- rbind(Results, Rndinfo);
    }
    Results <- insert_cell_history(Results);
    Results <- insert_neigh_fall(Results);
    namepos            <- which(dat == "HHID:") + 1;
    if(dat[namepos[1]] == 0){
        namepos <- which(dat == "Name:") + 1;
    }
    colnames(Results)  <- c("Round_number", dat[namepos]);
    game_type          <- strsplit(x = game_types, split = ",")[[1]][1];
    all_info           <- list(Game = game_type, Results = Results);
    return(all_info);
}

insert_neigh_fall  <- function(Results){
    max_round <- max(Results[,1]);
    for(j in 2:max_round){
        curr_rnd <- Results[Results[,1] == j,];
        last_rnd <- Results[Results[,1] == j-1,];
        for(i in 1:36){
            cell_neigh  <- paste("Cell_neig_",i, sep = ""); 
            cn_row      <- which(rownames(curr_rnd) == cell_neigh);
            neighs      <- neighbour_define(i);
            neigh_neigh <- paste("Cell_choice_",neighs, sep = "");
            cn_nn       <- which(rownames(last_rnd) %in% neigh_neigh);
            check_neigh <- last_rnd[cn_nn, 2:5];
            neigh_fall  <- dim(check_neigh)[1] - sum(check_neigh);
            Rloc        <- Results[,1] == j & rownames(Results) == cell_neigh;
            Results[Rloc, 2] <- neigh_fall;
        }
    }
    return(Results);
}

insert_cell_history <- function(Results){
    for(i in 1:36){
        cell_choice <- paste("Cell_choice_",i, sep = "");
        cell_hist   <- paste("Cell_hist_",i, sep = "");
        row_choice  <- which(rownames(Results) == cell_choice);
        row_hist    <- which(rownames(Results) == cell_hist);
        working_his <- matrix(data = 0, nrow = 4, ncol = 4);
        for(j in 2:length(row_hist)){
            the_choices <- Results[row_choice,];
            working_his <- rbind(working_his[,1:4], the_choices[j - 1, 2:5]);
            working_his <- working_his[2:5,];
            his_vec     <- cell_history_to_dec(t(working_his));
            Results[row_hist[j], 2:5] <- his_vec;
        }
    }
    return(Results);
}


neighbour_define <- function(cell){
    neighs <- NULL;
    if(cell == 1){
        neighs <- c(2, 7, 8);
    }
    if(cell == 2){
        neighs <- c(1, 3, 7, 8, 9);
    }
    if(cell == 3){
        neighs <- c(2, 4, 8, 9, 10);
    }
    if(cell == 4){
        neighs <- c(3, 9, 10, 11, 5);
    }
    if(cell == 5){
        neighs <- c(4, 10, 11, 12, 6);
    }
    if(cell == 6){
        neighs <- c(5, 11, 12);
    }
    if(cell == 7){
        neighs <- c(1, 2, 8, 13, 14);
    }
    if(cell == 8){
        neighs <- c(1, 2, 3, 7, 9, 13, 14, 15);
    }
    if(cell == 9){
        neighs <- c(2, 3, 4, 8, 10, 14, 15, 16);
    }
    if(cell == 10){
        neighs <- c(3, 4, 5, 9, 11, 15, 16, 17);
    }
    if(cell == 11){
        neighs <- c(4, 5, 6, 10, 12, 16, 17, 18);
    }
    if(cell == 12){
        neighs <- c(5, 6, 11, 17, 18);
    }
    if(cell == 13){
        neighs <- c(7, 8, 14, 19, 20);
    }
    if(cell == 14){
        neighs <- c(7, 8, 9, 13, 15, 19, 20, 21);
    }
    if(cell == 15){
        neighs <- c(8, 9, 10, 14, 16, 20, 21, 22);
    }
    if(cell == 16){
        neighs <- c(9, 10, 11, 15, 17, 21, 22, 23);
    }
    if(cell == 17){
        neighs <- c(10, 11, 12, 16, 18, 22, 23, 24);
    }
    if(cell == 18){
        neighs <- c(11, 12, 17, 23, 24);
    }
    if(cell == 19){
        neighs <- c(13, 14, 20, 25, 26);
    }
    if(cell == 20){
        neighs <- c(13, 14, 15, 19, 21, 25, 26, 27);
    }
    if(cell == 21){
        neighs <- c(14, 15, 16, 20, 22, 26, 27, 28);
    }
    if(cell == 22){
        neighs <- c(15, 16, 17, 21, 23, 27, 28, 29);
    }
    if(cell == 23){
        neighs <- c(16, 17, 18, 22, 24, 28, 29, 30);
    }
    if(cell == 24){
        neighs <- c(17, 18, 23, 29, 30);
    }
    if(cell == 25){
        neighs <- c(19, 20, 26, 31, 32);
    }
    if(cell == 26){
        neighs <- c(19, 20, 21, 25, 27, 31, 32, 33);
    }
    if(cell == 27){
        neighs <- c(20, 21, 22, 26, 28, 32, 33, 34);
    }
    if(cell == 28){
        neighs <- c(21, 22, 23, 27, 29, 33, 34, 35);
    }
    if(cell == 29){
        neighs <- c(22, 23, 24, 28, 30, 34, 35, 36);
    }
    if(cell == 30){
        neighs <- c(23, 24, 29, 35, 36);
    }
    if(cell == 31){
        neighs <- c(25, 26, 32);
    }
    if(cell == 32){
        neighs <- c(25, 26, 27, 31, 33);
    }
    if(cell == 33){
        neighs <- c(26, 27, 28, 32, 34);
    }
    if(cell == 34){
        neighs <- c(27, 28, 29, 33, 35);
    }
    if(cell == 35){
        neighs <- c(28, 29, 30, 34, 36);
    }
    if(cell == 36){
        neighs <- c(29, 30, 35);
    }
    return(neighs);
}


cell_history_to_dec <- function(cell_hist){
    hist_vec <- rep(0, times = dim(cell_hist)[1]);
    for(i in 1:dim(cell_hist)[1]){
        if(sum(cell_hist[i,] == c(0, 0, 0, 0)) == 4){
            hist_vec[i] <- 0;
        }
        if(sum(cell_hist[i,] == c(0, 0, 0, 1)) == 4){
            hist_vec[i] <- 1;
        }
        if(sum(cell_hist[i,] == c(0, 0, 1, 0)) == 4){
            hist_vec[i] <- 2;
        }
        if(sum(cell_hist[i,] == c(0, 0, 1, 1)) == 4){
            hist_vec[i] <- 3;
        }
        if(sum(cell_hist[i,] == c(0, 1, 0, 0)) == 4){
            hist_vec[i] <- 4;
        }
        if(sum(cell_hist[i,] == c(0, 1, 0, 1)) == 4){
            hist_vec[i] <- 5;
        }
        if(sum(cell_hist[i,] == c(0, 1, 1, 0)) == 4){
            hist_vec[i] <- 6;
        }
        if(sum(cell_hist[i,] == c(0, 1, 1, 1)) == 4){
            hist_vec[i] <- 7;
        }
        if(sum(cell_hist[i,] == c(1, 0, 0, 0)) == 4){
            hist_vec[i] <- 8;
        }
        if(sum(cell_hist[i,] == c(1, 0, 0, 1)) == 4){
            hist_vec[i] <- 9;
        }
        if(sum(cell_hist[i,] == c(1, 0, 1, 0)) == 4){
            hist_vec[i] <- 10;
        }
        if(sum(cell_hist[i,] == c(1, 0, 1, 1)) == 4){
            hist_vec[i] <- 11;
        }
        if(sum(cell_hist[i,] == c(1, 1, 0, 0)) == 4){
            hist_vec[i] <- 12;
        }
        if(sum(cell_hist[i,] == c(1, 1, 0, 1)) == 4){
            hist_vec[i] <- 13;
        }
        if(sum(cell_hist[i,] == c(1, 1, 1, 0)) == 4){
            hist_vec[i] <- 14;
        }
        if(sum(cell_hist[i,] == c(1, 1, 1, 1)) == 4){
            hist_vec[i] <- 15;
        }
    }
    return(hist_vec);
}

cell_summary <- function(choices, taps){
    mat1 <- matrix(data = 0, nrow = length(choices), ncol = 4);
    mat2 <- matrix(data = 0, nrow = length(choices), ncol = 4);
    for(i in 1:4){
        chc         <- which(choices == i);
        mat1[chc,i] <- 1;
        mat2[chc,i] <- taps[chc];
    }
    mat   <- rbind(mat1, mat2);
    chc_n <- paste("Cell_choice_", 1:36, sep = "");
    tap_n <- paste("Cell_tap", 1:36, sep = "");
    rownames(mat) <- c(chc_n, tap_n);
    return(mat);
}


get_subsidy_level <- function(dat){
    subsidy        <- which(dat == "Subsidy")[1];
    forest_subsidy <- as.numeric(dat[subsidy + 2]);
    return(forest_subsidy);
}

get_round_times <- function(dat){
    times      <- which(dat == "Time:");
    rnd_times  <- dat[times + 1];
    splt_times <- strsplit(x = rnd_times, split = ":");
    rnd_dates  <- dat[times + 3];
    splt_dates <- strsplit(x = rnd_dates, split = "-");
    num_dates  <- NULL;
    rounds     <- length(splt_dates);
    for(round in 1:rounds){
        new_date   <- make_month_numeric(splt_dates[[round]], position = 2);
        num_dates  <- rbind(num_dates, as.numeric(new_date))
    }
    rounds     <- length(splt_times);
    time_table <- matrix(data = 0, nrow = rounds, ncol = 3);
    for(round in 1:rounds){
        num_times          <- as.numeric(splt_times[[round]]);
        time_table[round,] <- num_times;
    }
    time_info           <- cbind(1:round, num_dates, time_table);
    colnames(time_info) <- c("Round", "Day", "Month", "Year", "Hour", "Minute", 
                             "Second");
    return(time_info);
}

make_month_numeric <- function(list, position = 2){
    if(list[position] == "Jan"){
        list[position] <- 1;
    }
    if(list[position] == "Feb"){
        list[position] <- 2;
    }
    if(list[position] == "Mar"){
        list[position] <- 3;
    }
    if(list[position] == "Apr"){
        list[position] <- 4;
    }
    if(list[position] == "May"){
        list[position] <- 5;
    }
    if(list[position] == "Jun"){
        list[position] <- 6;
    }
    if(list[position] == "Jul"){
        list[position] <- 7;
    }
    if(list[position] == "Aug"){
        list[position] <- 8;
    }
    if(list[position] == "Sep"){
        list[position] <- 9;
    }
    if(list[position] == "Oct"){
        list[position] <- 10;
    }
    if(list[position] == "Nov"){
        list[position] <- 11;
    }
    if(list[position] == "Dec"){
        list[position] <- 12;
    }
    return(list);
}

get_round_number <- function(dat, tags, which_tag = 1, use_choice = TRUE){
    if(use_choice == FALSE){
        tag_pos <- tags[which_tag];
        rnd_pos <- tag_pos + 11;
        if(dat[tag_pos + 8] != "Number" | dat[tag_pos + 10] != "Rounds:"){
            stop("I can't extract the round number for some reason");
        }
        extrRnd <- strsplit(x = dat[rnd_pos], split = "");
        rounds  <- as.numeric(extrRnd[[1]][1]);
    }else{
        choices <- which(dat == "Choices")
        rounds  <- length(choices);
    }
    return(rounds);
}

get_taps <- function(dat, round){
    choices <- which(dat == "Taps")[round];
    vals    <- dat[(choices+1):(choices+36)];
    extrVal <- strsplit(x = vals, split = "");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    cells   <- as.numeric(unlist(cleanEV));
    names(cells) <- paste("cell_", 1:36, sep = "");
    return(cells);
}

get_choices <- function(dat, round){
    choices <- which(dat == "Users")[round];
    vals    <- dat[(choices+1):(choices+36)];
    extrVal <- strsplit(x = vals, split = "");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    cleanEV[cleanEV == "-"] <- "-1";
    cells   <- as.numeric(unlist(cleanEV));
    names(cells) <- paste("cell_", 1:36, sep = "");
    return(cells);
}

get_scores  <- function(dat, round){
    sr      <- which(dat == "Summary:")[round];
    #--- Get yields
    yields  <- c(dat[sr+4], dat[sr+19], dat[sr+34], dat[sr+49]);
    extrVal <- strsplit(x = yields, split = ",");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    yieldsC <- as.numeric(unlist(cleanEV));
    #--- Get Bonuses
    bonus   <- c(dat[sr+6], dat[sr+21], dat[sr+36], dat[sr+51]);
    extrVal <- strsplit(x = bonus, split = ",");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    bonusC  <- as.numeric(unlist(cleanEV));
    #--- Get ES_Bump
    spend   <- c(dat[sr+9], dat[sr+24], dat[sr+39], dat[sr+54]);
    extrVal <- strsplit(x = spend, split = ",");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    es_bump <- as.numeric(unlist(cleanEV));
    #--- Get Round Score
    rscore  <- c(dat[sr+12], dat[sr+27], dat[sr+42], dat[sr+57]);
    extrVal <- strsplit(x = rscore, split = ",");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    rscoreC <- as.numeric(unlist(cleanEV));
    #--- Get Total Score
    rscore  <- c(dat[sr+15], dat[sr+30], dat[sr+45], dat[sr+60]);
    extrVal <- strsplit(x = rscore, split = ",");
    cleanEV <- lapply(extrVal, function(l) l[[1]])
    rscoreT <- as.numeric(unlist(cleanEV));
    #--- Clean it all up
    scores  <- rbind(yieldsC, bonusC, es_bump, rscoreC, rscoreT);
    return(scores);
}



